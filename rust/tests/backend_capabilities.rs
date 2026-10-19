use qps_inference::{
    BackendCapabilities, ContractValidationError, load_backend_capabilities,
    validate_backend_capabilities,
};
use serde_json::{Value, json};

const BACKEND_FIXTURE: &str = include_str!("../../contracts/fixtures/inference/rust-backend.json");

const ARRAY_FIELDS: [&str; 8] = [
    "manifestVersions",
    "weightFormats",
    "weightLayouts",
    "storedWeightDtypes",
    "modelProfiles",
    "promptProfiles",
    "executionProfiles",
    "generationModes",
];

fn fixture_document() -> Value {
    serde_json::from_str(BACKEND_FIXTURE).expect("Approved backend fixture must contain valid JSON")
}

fn backend_document(document: &Value) -> BackendCapabilities {
    serde_json::from_value(document.clone()).expect("Test backend descriptor must deserialize")
}

fn backend_with(pointer: &str, value: Value) -> BackendCapabilities {
    let mut document = fixture_document();
    *document
        .pointer_mut(pointer)
        .expect("Fixture field must exist") = value;
    backend_document(&document)
}

#[test]
fn approved_backend_fixture_deserializes_into_capabilities() {
    let backend: BackendCapabilities =
        serde_json::from_str(BACKEND_FIXTURE).expect("Approved backend fixture should deserialize");

    assert_eq!(backend.version, 1);
    assert_eq!(backend.id, "rust-qwen3");
    assert_eq!(backend.artifact_kind, "merged_model");
    assert_eq!(backend.manifest_versions, vec![1]);
    assert_eq!(backend.weight_formats.len(), 1);
    assert_eq!(backend.weight_layouts.len(), 1);
    assert_eq!(backend.stored_weight_dtypes.len(), 1);
    assert_eq!(backend.model_profiles, vec!["qwen3-0.6b-v1".to_owned()]);
    assert_eq!(
        backend.prompt_profiles,
        vec!["qwen3-chat-nonthinking-v1".to_owned()]
    );
    assert_eq!(backend.execution_profiles.len(), 1);
    assert_eq!(backend.execution_profiles[0].tensor_parallel_size, 1);
    assert_eq!(backend.generation_modes.len(), 1);
    assert_eq!(backend.max_context_tokens, 2048);
    assert_eq!(backend.max_output_tokens, 512);
    assert_eq!(backend.max_active_generations, 1);
    assert!(backend.streaming);
    assert_eq!(backend.http_contract, "qps-openai-subset-v1");

    validate_backend_capabilities(&backend).expect("Approved backend fixture must validate");
}

#[test]
fn embedded_backend_loader_returns_valid_capabilities() {
    let backend = load_backend_capabilities().expect("Embedded backend descriptor must load");

    assert_eq!(backend.id, "rust-qwen3");
    assert_eq!(backend.model_profiles, vec!["qwen3-0.6b-v1"]);
    validate_backend_capabilities(&backend).expect("Embedded backend descriptor must validate");
}

#[test]
fn missing_required_fields_are_rejected_during_parsing() {
    for field in fixture_document().as_object().unwrap().keys() {
        let mut document = fixture_document();
        document.as_object_mut().unwrap().remove(field);

        let error = serde_json::from_value::<BackendCapabilities>(document)
            .expect_err("Missing required backend field must fail parsing");
        assert!(error.is_data(), "Missing {field}: {error}");
        assert!(
            error.to_string().contains(field),
            "Missing {field}: {error}"
        );
    }

    for field in ["device", "dtype", "tensorParallelSize"] {
        let mut document = fixture_document();
        document["executionProfiles"][0]
            .as_object_mut()
            .unwrap()
            .remove(field);

        let error = serde_json::from_value::<BackendCapabilities>(document)
            .expect_err("Missing required execution-profile field must fail parsing");
        assert!(error.is_data(), "Missing {field}: {error}");
        assert!(
            error.to_string().contains(field),
            "Missing {field}: {error}"
        );
    }
}

#[test]
fn unknown_fields_are_rejected_at_root_and_inside_execution_profiles() {
    for pointer in ["", "/executionProfiles/0"] {
        let mut document = fixture_document();
        document
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unexpectedField".into(), json!(true));

        let error = serde_json::from_value::<BackendCapabilities>(document)
            .expect_err("Unknown backend fields must fail parsing");
        assert!(error.is_data(), "At {pointer}: {error}");
        assert!(error.to_string().contains("unexpectedField"));
    }
}

#[test]
fn invalid_enum_values_are_rejected_during_parsing() {
    for (pointer, value) in [
        ("/weightFormats/0", "pytorch"),
        ("/weightLayouts/0", "singleFile"),
        ("/storedWeightDtypes/0", "fp16"),
        ("/executionProfiles/0/device", "mps"),
        ("/executionProfiles/0/dtype", "fp16"),
        ("/generationModes/0", "sampling"),
    ] {
        let mut document = fixture_document();
        *document.pointer_mut(pointer).unwrap() = json!(value);

        let error = serde_json::from_value::<BackendCapabilities>(document)
            .expect_err("Unrecognized enum values must fail before value validation");
        assert!(error.is_data(), "At {pointer}: {error}");
        assert!(error.to_string().contains(value), "At {pointer}: {error}");
    }
}

#[test]
fn unsupported_descriptor_versions_report_the_supplied_version() {
    for version in [0, 2, u32::MAX] {
        let backend = backend_with("/version", json!(version));

        match validate_backend_capabilities(&backend) {
            Err(ContractValidationError::UnsupportedVersion { found, expected }) => {
                assert_eq!(found, version);
                assert_eq!(expected, 1);
            }
            other => panic!("Expected unsupported descriptor version {version}, got: {other:?}"),
        }
    }
}

#[test]
fn backend_and_profile_identifiers_reject_invalid_formats() {
    for (pointer, field) in [
        ("/id", "backend_capability.id"),
        ("/modelProfiles/0", "backend_capability.modelProfiles"),
        ("/promptProfiles/0", "backend_capability.promptProfiles"),
    ] {
        for invalid in [
            "".to_owned(),
            "Qwen".to_owned(),
            "1qwen".to_owned(),
            "qwen_model".to_owned(),
            "qwen model".to_owned(),
            "qwén".to_owned(),
            "q".repeat(81),
        ] {
            let backend = backend_with(pointer, json!(invalid));

            match validate_backend_capabilities(&backend) {
                Err(ContractValidationError::UnsupportedProfile { profile, found, .. }) => {
                    assert_eq!(profile, field);
                    assert_eq!(found, invalid);
                }
                other => panic!("Expected invalid identifier at {pointer}, got: {other:?}"),
            }
        }
    }
}

#[test]
fn backend_and_profile_identifiers_accept_boundary_lengths_and_allowed_characters() {
    for pointer in ["/id", "/modelProfiles/0", "/promptProfiles/0"] {
        for valid in ["q".to_owned(), "qwen3.model-v1".to_owned(), "q".repeat(80)] {
            let backend = backend_with(pointer, json!(valid));
            validate_backend_capabilities(&backend)
                .unwrap_or_else(|error| panic!("Valid {pointer}={valid:?} failed: {error:?}"));
        }
    }
}

#[test]
fn unsupported_artifact_and_http_contracts_report_the_correct_field() {
    for (pointer, field, supplied, expected_contract) in [
        (
            "/artifactKind",
            "backend_capability.artifactKind",
            "adapter",
            "merged_model",
        ),
        (
            "/httpContract",
            "backend_capability.httpContract",
            "openai-v1",
            "qps-openai-subset-v1",
        ),
    ] {
        let backend = backend_with(pointer, json!(supplied));

        match validate_backend_capabilities(&backend) {
            Err(ContractValidationError::InvalidStringValue {
                value,
                found,
                expected,
            }) => {
                assert_eq!(value, field);
                assert_eq!(found, supplied);
                assert_eq!(expected, expected_contract);
            }
            other => panic!("Expected unsupported contract at {pointer}, got: {other:?}"),
        }
    }
}

#[test]
fn serving_limits_accept_inclusive_bounds_and_positive_concurrency() {
    for (pointer, values) in [
        ("/maxContextTokens", [16, 2048]),
        ("/maxOutputTokens", [1, 512]),
        ("/maxActiveGenerations", [1, u32::MAX]),
    ] {
        for value in values {
            let backend = backend_with(pointer, json!(value));
            validate_backend_capabilities(&backend)
                .unwrap_or_else(|error| panic!("Valid {pointer}={value} failed: {error:?}"));
        }
    }
}

#[test]
fn invalid_serving_limits_report_the_correct_field_value_and_bounds() {
    for (pointer, field, values, expected_bounds) in [
        (
            "/maxContextTokens",
            "backend_capability.maxContextTokens",
            vec![0, 15, 2049],
            "16 through 2048, inclusive",
        ),
        (
            "/maxOutputTokens",
            "backend_capability.maxOutputTokens",
            vec![0, 513],
            "1 through 512, inclusive",
        ),
        (
            "/maxActiveGenerations",
            "backend_capability.maxActiveGenerations",
            vec![0],
            "greater than 0",
        ),
    ] {
        for supplied in values {
            let backend = backend_with(pointer, json!(supplied));

            match validate_backend_capabilities(&backend) {
                Err(ContractValidationError::InvalidNumericValue {
                    value,
                    found,
                    expected,
                }) => {
                    assert_eq!(value, field);
                    assert_eq!(found, supplied);
                    assert_eq!(expected, expected_bounds);
                }
                other => panic!("Expected invalid limit at {pointer}, got: {other:?}"),
            }
        }
    }
}

#[test]
fn every_capability_list_must_be_nonempty() {
    for field in ARRAY_FIELDS {
        let backend = backend_with(&format!("/{field}"), json!([]));

        match validate_backend_capabilities(&backend) {
            Err(ContractValidationError::EmptyList { field: reported }) => {
                assert_eq!(reported, format!("backend_capability.{field}"));
            }
            other => panic!("Expected empty-list error for {field}, got: {other:?}"),
        }
    }
}

#[test]
fn every_capability_list_rejects_duplicate_values() {
    for field in ARRAY_FIELDS {
        let mut document = fixture_document();
        let item = document[field][0].clone();
        document[field] = json!([item.clone(), item]);

        match validate_backend_capabilities(&backend_document(&document)) {
            Err(ContractValidationError::DuplicateEntries { field: reported }) => {
                assert_eq!(reported, format!("backend_capability.{field}"));
            }
            other => panic!("Expected duplicate-list error for {field}, got: {other:?}"),
        }
    }
}

#[test]
fn nonadjacent_duplicates_are_rejected() {
    for (field, entries) in [
        ("storedWeightDtypes", json!(["fp32", "bf16", "fp32"])),
        ("modelProfiles", json!(["model-a", "model-b", "model-a"])),
        (
            "promptProfiles",
            json!(["prompt-a", "prompt-b", "prompt-a"]),
        ),
        (
            "executionProfiles",
            json!([
                {"device": "cuda", "dtype": "bf16", "tensorParallelSize": 1},
                {"device": "cpu", "dtype": "fp32", "tensorParallelSize": 1},
                {"device": "cuda", "dtype": "bf16", "tensorParallelSize": 1}
            ]),
        ),
    ] {
        let backend = backend_with(&format!("/{field}"), entries);

        match validate_backend_capabilities(&backend) {
            Err(ContractValidationError::DuplicateEntries { field: reported }) => {
                assert_eq!(reported, format!("backend_capability.{field}"));
            }
            other => panic!("Expected nonadjacent duplicate in {field}, got: {other:?}"),
        }
    }
}

#[test]
fn unsupported_manifest_versions_report_the_list_item_not_the_descriptor_version() {
    for version in [0, 2, u32::MAX] {
        let backend = backend_with("/manifestVersions", json!([version]));
        assert_eq!(backend.version, 1);

        match validate_backend_capabilities(&backend) {
            Err(ContractValidationError::UnsupportedVersion { found, expected }) => {
                assert_eq!(found, version);
                assert_eq!(expected, 1);
            }
            other => panic!("Expected unsupported manifest version {version}, got: {other:?}"),
        }
    }
}

#[test]
fn distinct_profiles_and_stored_dtypes_are_valid_descriptor_values() {
    let mut document = fixture_document();
    document["storedWeightDtypes"] = json!(["fp32", "bf16"]);
    document["modelProfiles"] = json!(["qwen3-0.6b-v1", "another-model-v1"]);
    document["promptProfiles"] = json!(["qwen3-chat-nonthinking-v1", "another-prompt-v1"]);
    document["executionProfiles"] = json!([
        {"device": "cuda", "dtype": "bf16", "tensorParallelSize": 1},
        {"device": "cpu", "dtype": "bf16", "tensorParallelSize": 1},
        {"device": "cuda", "dtype": "fp32", "tensorParallelSize": 1},
        {"device": "cuda", "dtype": "bf16", "tensorParallelSize": 2}
    ]);

    // Descriptor validity is separate from matching a request to this engine's capabilities.
    validate_backend_capabilities(&backend_document(&document))
        .expect("Distinct valid values must not be treated as duplicates or unsupported requests");
}

#[test]
fn invalid_tensor_parallel_sizes_report_the_exact_execution_profile_index() {
    let valid_profiles = json!([
        {"device": "cuda", "dtype": "bf16", "tensorParallelSize": 1},
        {"device": "cuda", "dtype": "bf16", "tensorParallelSize": 2}
    ]);

    for index in 0..=2 {
        for supplied in [0, 3, u32::MAX] {
            let mut profiles = valid_profiles.as_array().unwrap()[..index].to_vec();
            profiles.push(json!({
                "device": "cuda", "dtype": "bf16", "tensorParallelSize": supplied
            }));
            let backend = backend_with("/executionProfiles", json!(profiles));

            match validate_backend_capabilities(&backend) {
                Err(ContractValidationError::InvalidNumericValue {
                    value,
                    found,
                    expected,
                }) => {
                    assert_eq!(
                        value,
                        format!("backend_capability.executionProfiles[{index}].tensorParallelSize")
                    );
                    assert_eq!(found, supplied);
                    assert_eq!(expected, "1 or 2");
                }
                other => panic!("Expected invalid TP size at index {index}, got: {other:?}"),
            }
        }
    }
}

#[test]
fn streaming_can_be_advertised_as_enabled_or_disabled() {
    for enabled in [true, false] {
        let backend = backend_with("/streaming", json!(enabled));
        assert_eq!(backend.streaming, enabled);
        validate_backend_capabilities(&backend)
            .expect("Either boolean is valid in a capability descriptor");
    }
}
