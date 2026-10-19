use serde_json::from_str;

use crate::{
    BackendCapabilities, Runtime,
    contracts::{Dtype, GenerationMode, WeightFormat, WeightLayout},
};
use std::{
    collections::HashSet,
    format,
    fs::read_to_string,
    matches,
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
};

const BACKEND_JSON: &str = include_str!("../../contracts/fixtures/inference/rust-backend.json");

#[derive(Debug)]
pub enum ContractLoadError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: Option<PathBuf>,
        source: serde_json::Error,
    },
}

#[derive(Debug)]
pub enum ContractValidationError {
    UnsupportedVersion {
        found: u32,
        expected: u32,
    },
    UnsupportedIdentity {
        identity: &'static str,
        found: String,
        expected: &'static str,
    },
    UnsupportedProfile {
        profile: &'static str,
        found: String,
        expected: &'static str,
    },
    UnsupportedServedModelName {
        found: String,
        expected: &'static str,
    },
    UnsupportedHash {
        hash: &'static str,
        found: String,
        expected: &'static str,
    },
    InvalidArtifactDirectory {
        found: PathBuf,
        expected: &'static str,
    },
    InvalidNumericValue {
        value: String,
        found: u32,
        expected: &'static str,
    },
    InvalidStringValue {
        value: &'static str,
        found: String,
        expected: &'static str,
    },
    InvalidBindAddress {
        found: IpAddr,
        expected: IpAddr,
    },
    EmptyList {
        field: &'static str,
    },
    DuplicateEntries {
        field: &'static str,
    },
}

pub fn read_runtime(file_path: &Path) -> Result<Runtime, ContractLoadError> {
    let file_str = read_to_string(file_path).map_err(|source| ContractLoadError::Read {
        path: file_path.to_path_buf(),
        source,
    })?;

    let runtime: Runtime = from_str(&file_str).map_err(|source| ContractLoadError::Parse {
        path: Some(file_path.to_path_buf()),
        source,
    })?;

    Ok(runtime)
}

fn is_valid_resource_id(value: &str, prefix: &str) -> bool {
    let Some(suffix) = value.strip_prefix(prefix) else {
        return false;
    };

    suffix.len() == 32
        && suffix
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_valid_profile(value: &str) -> bool {
    let mut bytes = value.bytes();

    let Some(first) = bytes.next() else {
        return false;
    };

    value.len() <= 80
        && first.is_ascii_lowercase()
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-')
        })
}

fn is_valid_served_model_name(value: &str) -> bool {
    let mut bytes = value.bytes();

    let Some(first) = bytes.next() else {
        return false;
    };

    value.len() <= 80
        && first.is_ascii_alphanumeric()
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

fn is_valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn validate_item_list<T, F>(
    items: &[T],
    field: &'static str,
    validate_item: F,
) -> Result<(), ContractValidationError>
where
    T: Eq + std::hash::Hash,
    F: Fn(usize, &T) -> Result<(), ContractValidationError>,
{
    if items.is_empty() {
        return Err(ContractValidationError::EmptyList { field });
    }

    let mut seen = HashSet::new();

    for (index, item) in items.iter().enumerate() {
        if !seen.insert(item) {
            return Err(ContractValidationError::DuplicateEntries { field });
        }

        validate_item(index, item)?
    }

    Ok(())
}

pub fn validate_runtime(runtime: &Runtime) -> Result<(), ContractValidationError> {
    if runtime.version != 1 {
        return Err(ContractValidationError::UnsupportedVersion {
            found: runtime.version,
            expected: 1,
        });
    }

    if !is_valid_resource_id(&runtime.deployment_id, "dep-") {
        return Err(ContractValidationError::UnsupportedIdentity {
            identity: "runtime.deploymentId",
            found: runtime.deployment_id.clone(),
            expected: "dep- followed by 32 lowercase hexadecimal characters",
        });
    }

    if !is_valid_resource_id(&runtime.model_version_id, "model-") {
        return Err(ContractValidationError::UnsupportedIdentity {
            identity: "runtime.modelVersionId",
            found: runtime.model_version_id.clone(),
            expected: "model- followed by 32 lowercase hexadecimal characters",
        });
    }

    if !is_valid_resource_id(&runtime.artifact.id, "art-") {
        return Err(ContractValidationError::UnsupportedIdentity {
            identity: "runtime.artifact.id",
            found: runtime.artifact.id.clone(),
            expected: "art- followed by 32 lowercase hexadecimal characters",
        });
    }

    if !is_valid_profile(&runtime.backend) {
        return Err(ContractValidationError::UnsupportedProfile {
            profile: "runtime.backend",
            found: runtime.backend.clone(),
            expected: "1-80 ASCII characters; first a-z; remaining a-z, 0-9, '.' or '-'",
        });
    }

    if !is_valid_profile(&runtime.model_profile) {
        return Err(ContractValidationError::UnsupportedProfile {
            profile: "runtime.modelProfile",
            found: runtime.model_profile.clone(),
            expected: "1-80 ASCII characters; first a-z; remaining a-z, 0-9, '.' or '-'",
        });
    }

    if !is_valid_served_model_name(&runtime.served_model_name) {
        return Err(ContractValidationError::UnsupportedServedModelName {
            found: runtime.served_model_name.clone(),
            expected: "1-80 ASCII characters; first a letter or digit; remaining letters, digits, '_', '.' or '-'",
        });
    }

    if !is_valid_hash(&runtime.artifact.manifest_sha) {
        return Err(ContractValidationError::UnsupportedHash {
            hash: "runtime.artifact.manifestSha",
            found: runtime.artifact.manifest_sha.clone(),
            expected: "exactly 64 lowercase hexadecimal characters (0-9, a-f)",
        });
    }

    if !runtime.artifact.directory.is_absolute() {
        return Err(ContractValidationError::InvalidArtifactDirectory {
            found: runtime.artifact.directory.clone(),
            expected: "an absolute POSIX path",
        });
    }

    if ![1, 2].contains(&runtime.execution.tensor_parallel_size) {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "runtime.execution.tensorParallelSize".into(),
            found: runtime.execution.tensor_parallel_size,
            expected: "1 or 2",
        });
    }

    if !(16..=2048).contains(&runtime.limits.max_context_tokens) {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "runtime.limits.maxContextTokens".into(),
            found: runtime.limits.max_context_tokens,
            expected: "16 through 2048, inclusive",
        });
    }

    if !(1..=512).contains(&runtime.limits.max_output_tokens) {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "runtime.limits.maxOutputTokens".into(),
            found: runtime.limits.max_output_tokens,
            expected: "1 through 512, inclusive",
        });
    }

    if runtime.limits.max_active_generations == 0 {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "runtime.limits.maxActiveGenerations".into(),
            found: runtime.limits.max_active_generations,
            expected: "greater than 0",
        });
    }

    if runtime.http.port == 0 {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "runtime.http.port".into(),
            found: u32::from(runtime.http.port),
            expected: "greater than 0",
        });
    }

    let expected_bind = IpAddr::V4(Ipv4Addr::LOCALHOST);

    if runtime.http.bind != expected_bind {
        return Err(ContractValidationError::InvalidBindAddress {
            found: runtime.http.bind,
            expected: expected_bind,
        });
    }
    Ok(())
}

pub fn load_backend_capabilities() -> Result<BackendCapabilities, ContractLoadError> {
    let backend_capabilities =
        from_str(BACKEND_JSON).map_err(|source| ContractLoadError::Parse { path: None, source })?;

    Ok(backend_capabilities)
}

pub fn validate_backend_capabilities(
    backend_capability: &BackendCapabilities,
) -> Result<(), ContractValidationError> {
    if backend_capability.version != 1 {
        return Err(ContractValidationError::UnsupportedVersion {
            found: backend_capability.version,
            expected: 1,
        });
    }

    if !is_valid_profile(&backend_capability.id) {
        return Err(ContractValidationError::UnsupportedProfile {
            profile: "backend_capability.id",
            found: backend_capability.id.clone(),
            expected: "1-80 ASCII characters; first a-z; remaining a-z, 0-9, '.' or '-'",
        });
    }

    if backend_capability.artifact_kind != "merged_model" {
        return Err(ContractValidationError::InvalidStringValue {
            value: "backend_capability.artifactKind",
            found: backend_capability.artifact_kind.clone(),
            expected: "merged_model",
        });
    }

    if backend_capability.http_contract != "qps-openai-subset-v1" {
        return Err(ContractValidationError::InvalidStringValue {
            value: "backend_capability.httpContract",
            found: backend_capability.http_contract.clone(),
            expected: "qps-openai-subset-v1",
        });
    }

    if !(16..=2048).contains(&backend_capability.max_context_tokens) {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "backend_capability.maxContextTokens".into(),
            found: backend_capability.max_context_tokens,
            expected: "16 through 2048, inclusive",
        });
    }

    if !(1..=512).contains(&backend_capability.max_output_tokens) {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "backend_capability.maxOutputTokens".into(),
            found: backend_capability.max_output_tokens,
            expected: "1 through 512, inclusive",
        });
    }

    if backend_capability.max_active_generations == 0 {
        return Err(ContractValidationError::InvalidNumericValue {
            value: "backend_capability.maxActiveGenerations".into(),
            found: backend_capability.max_active_generations,
            expected: "greater than 0",
        });
    }

    validate_item_list(
        &backend_capability.manifest_versions,
        "backend_capability.manifestVersions",
        |_, version| {
            if *version != 1 {
                return Err(ContractValidationError::UnsupportedVersion {
                    found: *version,
                    expected: 1,
                });
            }
            Ok(())
        },
    )?;

    validate_item_list(
        &backend_capability.weight_formats,
        "backend_capability.weightFormats",
        |_, weight_format| match *weight_format {
            WeightFormat::Safetensors => Ok(()),
        },
    )?;

    validate_item_list(
        &backend_capability.weight_layouts,
        "backend_capability.weightLayouts",
        |_, weight_layout| match *weight_layout {
            WeightLayout::SingleFile => Ok(()),
        },
    )?;

    validate_item_list(
        &backend_capability.stored_weight_dtypes,
        "backend_capability.storedWeightDtypes",
        |_, stored_weight_dtype| match *stored_weight_dtype {
            Dtype::Bf16 => Ok(()),
            Dtype::Fp32 => Ok(()),
        },
    )?;

    validate_item_list(
        &backend_capability.model_profiles,
        "backend_capability.modelProfiles",
        |_, model_profile| {
            if !is_valid_profile(model_profile) {
                return Err(ContractValidationError::UnsupportedProfile {
                    profile: "backend_capability.modelProfiles",
                    found: model_profile.clone(),
                    expected: "1-80 ASCII characters; first a-z; remaining a-z, 0-9, '.' or '-'",
                });
            }

            Ok(())
        },
    )?;

    validate_item_list(
        &backend_capability.prompt_profiles,
        "backend_capability.promptProfiles",
        |_, prompt_profile| {
            if !is_valid_profile(prompt_profile) {
                return Err(ContractValidationError::UnsupportedProfile {
                    profile: "backend_capability.promptProfiles",
                    found: prompt_profile.clone(),
                    expected: "1-80 ASCII characters; first a-z; remaining a-z, 0-9, '.' or '-'",
                });
            }

            Ok(())
        },
    )?;

    validate_item_list(
        &backend_capability.execution_profiles,
        "backend_capability.executionProfiles",
        |idx, execution_profile| {
            if ![1, 2].contains(&execution_profile.tensor_parallel_size) {
                return Err(ContractValidationError::InvalidNumericValue {
                    value: format!(
                        "backend_capability.executionProfiles[{idx}].tensorParallelSize"
                    ),
                    found: execution_profile.tensor_parallel_size,
                    expected: "1 or 2",
                });
            }

            Ok(())
        },
    )?;

    validate_item_list(
        &backend_capability.generation_modes,
        "backend_capability.generationModes",
        |_, generation_mode| match *generation_mode {
            GenerationMode::Greedy => Ok(()),
        },
    )?;

    Ok(())
}
