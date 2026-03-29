use std::convert::Infallible;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use candle_core::Device;
use http::header::{HeaderValue, CONTENT_TYPE};
use http::{Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use turbocalm_models::CalmAutoencoderConfig;

use crate::embedding::{token_count_for_text, EmbeddingMode, EmbeddingModel};

pub const DEFAULT_MODEL_NAME: &str = "turbocalm-local";

#[derive(Clone)]
pub struct ServerState {
    pub model: Arc<EmbeddingModel>,
    pub mode: EmbeddingMode,
    pub model_name: Arc<str>,
}

impl ServerState {
    pub fn new(model: EmbeddingModel, mode: EmbeddingMode) -> Self {
        Self {
            model: Arc::new(model),
            mode,
            model_name: Arc::from(DEFAULT_MODEL_NAME),
        }
    }
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

pub fn load_model_or_random(
    checkpoint_path: Option<&Path>,
    config: CalmAutoencoderConfig,
    device: Device,
) -> Result<EmbeddingModel> {
    if let Some(path) = checkpoint_path {
        match EmbeddingModel::from_checkpoint(path, config.clone(), device.clone()) {
            Ok(model) => return Ok(model),
            Err(error) => {
                eprintln!(
                    "warning: failed to load checkpoint {}; falling back to random weights: {error:#}",
                    path.display()
                );
            }
        }
    }

    EmbeddingModel::random(config, device)
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

    match build_response(state, request) {
        Ok(response) => Ok(json_response(StatusCode::OK, &response)),
        Err((status, error)) => Ok(error_response(status, error)),
    }
}

fn build_response(
    state: ServerState,
    request: EmbeddingRequest,
) -> std::result::Result<EmbeddingResponse, (StatusCode, String)> {
    if request.model != state.model_name.as_ref() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "unsupported model {:?}; expected {:?}",
                request.model, state.model_name
            ),
        ));
    }

    let texts = request.input.into_texts();
    if texts.is_empty() {
        return Err((
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
            .map(|embeddings| {
                embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| EmbeddingData {
                        embedding: EmbeddingPayload::Pooled(embedding),
                        index,
                    })
                    .collect()
            })
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")))?,
        EmbeddingMode::Chunked => state
            .model
            .embed_texts_chunked(&texts)
            .map(|embeddings| {
                embeddings
                    .into_iter()
                    .enumerate()
                    .map(|(index, embedding)| EmbeddingData {
                        embedding: EmbeddingPayload::Chunked(embedding),
                        index,
                    })
                    .collect()
            })
            .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, format!("{error:#}")))?,
    };

    Ok(EmbeddingResponse {
        data,
        model: state.model_name.to_string(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use http_body_util::Full;
    use hyper::client::conn::http1::handshake;
    use serde_json::json;
    use tokio::io::duplex;

    #[tokio::test]
    async fn in_memory_http_returns_128_dim_embedding() -> Result<()> {
        let config = CalmAutoencoderConfig {
            vocab_size: 512,
            hidden_size: 32,
            intermediate_size: 64,
            latent_size: 128,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            patch_size: 4,
            max_position_embeddings: 128,
            tie_word_embeddings: true,
            ..Default::default()
        };
        let model = EmbeddingModel::random(config, Device::Cpu)?;
        let state = ServerState::new(model, EmbeddingMode::Pooled);

        let (client_io, server_io) = duplex(1 << 16);
        let server_state = state.clone();
        let server = tokio::spawn(async move {
            let service = service_fn(move |request| handle_request(request, server_state.clone()));
            http1::Builder::new()
                .serve_connection(TokioIo::new(server_io), service)
                .await
        });

        let (mut sender, connection) = handshake(TokioIo::new(client_io)).await?;
        let client = tokio::spawn(async move { connection.await });

        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header(CONTENT_TYPE, "application/json")
            .body(Full::new(Bytes::from(serde_json::to_vec(&json!({
                "input": "steady breathing practice",
                "model": DEFAULT_MODEL_NAME,
            }))?)))?;

        let response = sender.send_request(request).await?;
        let body = response.into_body().collect().await?.to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body)?;
        let embedding = json["data"][0]["embedding"]
            .as_array()
            .context("expected pooled embedding array")?;
        assert_eq!(embedding.len(), 128);

        drop(sender);
        client.await??;
        server.await??;
        Ok(())
    }
}
