use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use http::header::{CONTENT_TYPE, HeaderValue};
use http::{Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

use crate::embedding::{EmbeddingMode, EmbeddingModel, token_count_for_text};

pub const DEFAULT_MODEL_NAME: &str = "turbocalm-local";

#[derive(Clone)]
pub struct ServerState {
    pub model: Arc<EmbeddingModel>,
    pub mode: EmbeddingMode,
    pub model_name: Arc<str>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    pub model: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    fn into_texts(self) -> Vec<String> {
        match self {
            Self::Single(text) => vec![text],
            Self::Batch(texts) => texts,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub embedding: EmbeddingPayload,
    pub index: usize,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingPayload {
    Pooled(Vec<f32>),
    Chunked(Vec<Vec<f32>>),
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

pub async fn serve_with_listener<F>(
    listener: TcpListener,
    state: ServerState,
    shutdown: F,
) -> Result<()>
where
    F: Future<Output = ()> + Send + 'static,
{
    let mut shutdown = std::pin::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let (stream, _) = accepted.context("failed to accept embedding connection")?;
                let state = state.clone();
                tokio::spawn(async move {
                    let io = TokioIo::new(stream);
                    let service = service_fn(move |request| handle_request(request, state.clone()));
                    if let Err(error) = http1::Builder::new().serve_connection(io, service).await {
                        eprintln!("embedding server connection error: {error}");
                    }
                });
            }
        }
    }
    Ok(())
}

pub async fn serve(state: ServerState, port: u16) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("failed to bind embedding server on port {port}"))?;
    let addr = listener
        .local_addr()
        .context("failed to read embedding server socket addr")?;
    println!("embedding server listening on http://{addr}/v1/embeddings");
    serve_with_listener(listener, state, async {})
        .await
        .context("failed to run embedding server")
}

async fn handle_request(
    request: Request<Incoming>,
    state: ServerState,
) -> Result<Response<Full<Bytes>>, Infallible> {
    if request.method() != Method::POST {
        return Ok(error_response(
            StatusCode::METHOD_NOT_ALLOWED,
            "only POST is supported".to_string(),
        ));
    }
    if request.uri().path() != "/v1/embeddings" {
        return Ok(error_response(
            StatusCode::NOT_FOUND,
            "not found".to_string(),
        ));
    }

    let body = match request.into_body().collect().await {
        Ok(body) => body.to_bytes(),
        Err(error) => {
            return Ok(error_response(
                StatusCode::BAD_REQUEST,
                format!("failed to read request body: {error}"),
            ));
        }
    };

    let request = match serde_json::from_slice::<EmbeddingRequest>(&body) {
        Ok(request) => request,
        Err(error) => {
            return Ok(error_response(
                StatusCode::BAD_REQUEST,
                format!("failed to parse request body: {error}"),
            ));
        }
    };

    if request.model != state.model_name.as_ref() {
        return Ok(error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported model {:?}; expected {:?}",
                request.model, state.model_name
            ),
        ));
    }

    let texts = request.input.into_texts();
    if texts.is_empty() {
        return Ok(error_response(
            StatusCode::BAD_REQUEST,
            "input must contain at least one text".to_string(),
        ));
    }

    let prompt_tokens = texts
        .iter()
        .map(|text| token_count_for_text(text, state.model.config()))
        .sum();

    let data = match state.mode {
        EmbeddingMode::Pooled => state
            .model
            .embed_texts_pooled(&texts)
            .map_err(internal_error)
            .map_or_else(
                |response| Err(response),
                |embeddings| {
                    Ok(embeddings
                        .into_iter()
                        .enumerate()
                        .map(|(index, embedding)| EmbeddingData {
                            embedding: EmbeddingPayload::Pooled(embedding),
                            index,
                        })
                        .collect())
                },
            ),
        EmbeddingMode::Chunked => state
            .model
            .embed_texts_chunked(&texts)
            .map_err(internal_error)
            .map_or_else(
                |response| Err(response),
                |embeddings| {
                    Ok(embeddings
                        .into_iter()
                        .enumerate()
                        .map(|(index, embedding)| EmbeddingData {
                            embedding: EmbeddingPayload::Chunked(embedding),
                            index,
                        })
                        .collect())
                },
            ),
    };

    let data = match data {
        Ok(data) => data,
        Err(response) => return Ok(response),
    };

    Ok(json_response(
        StatusCode::OK,
        &EmbeddingResponse {
        data,
        model: state.model_name.to_string(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    },
    ))
}

fn internal_error(error: anyhow::Error) -> Response<Full<Bytes>> {
    error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}"))
}

fn error_response(status: StatusCode, error: String) -> Response<Full<Bytes>> {
    json_response(status, &ErrorResponse { error })
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Response<Full<Bytes>> {
    let body = serde_json::to_vec(value).unwrap_or_else(|error| {
        format!(r#"{{"error":"failed to serialize response: {error}"}}"#).into_bytes()
    });
    let mut response = Response::new(Full::new(Bytes::from(body)));
    *response.status_mut() = status;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    response
}

pub fn resolve_mode(pooled: bool, chunked: bool) -> Result<EmbeddingMode> {
    if pooled && chunked {
        bail!("--pooled and --chunked are mutually exclusive")
    }
    if chunked {
        Ok(EmbeddingMode::Chunked)
    } else {
        Ok(EmbeddingMode::Pooled)
    }
}
