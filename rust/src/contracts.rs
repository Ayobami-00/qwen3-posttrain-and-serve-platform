use serde::Deserialize;
use std::{net::IpAddr, path::PathBuf};

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Runtime {
    pub version: u32,
    pub deployment_id: String,
    pub model_version_id: String,
    pub served_model_name: String,
    pub backend: String,
    pub model_profile: String,

    pub artifact: ArtifactRef,
    pub execution: ExecutionConfig,
    pub generation: GenerationConfig,
    pub limits: ServingLimits,
    pub http: HttpConfig,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactRef {
    pub id: String,
    pub directory: PathBuf,
    pub manifest_sha: String,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub enum Device {
    Cuda,
    Cpu,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub enum Dtype {
    Bf16,
    Fp32,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionConfig {
    pub device: Device,
    pub dtype: Dtype,
    pub tensor_parallel_size: u32,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub enum GenerationMode {
    Greedy,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GenerationConfig {
    pub mode: GenerationMode,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ServingLimits {
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
    pub max_active_generations: u32,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HttpConfig {
    pub bind: IpAddr,
    pub port: u16,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub enum WeightFormat {
    Safetensors,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum WeightLayout {
    SingleFile,
}

#[derive(Deserialize, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionProfile {
    pub device: Device,
    pub dtype: Dtype,
    pub tensor_parallel_size: u32,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BackendCapabilities {
    pub version: u32,
    pub id: String,
    pub artifact_kind: String,
    pub manifest_versions: Vec<u32>,
    pub weight_formats: Vec<WeightFormat>,
    pub weight_layouts: Vec<WeightLayout>,
    pub stored_weight_dtypes: Vec<Dtype>,
    pub model_profiles: Vec<String>,
    pub prompt_profiles: Vec<String>,
    pub execution_profiles: Vec<ExecutionProfile>,
    pub generation_modes: Vec<GenerationMode>,
    pub max_context_tokens: u32,
    pub max_output_tokens: u32,
    pub max_active_generations: u32,
    pub streaming: bool,
    pub http_contract: String,
}
