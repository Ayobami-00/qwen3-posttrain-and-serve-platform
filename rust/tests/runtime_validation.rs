use qps_inference::{
    ContractLoadError, ContractValidationError, Runtime, read_runtime, validate_runtime,
};
use serde_json::Value;
use std::{
    fs,
    io::ErrorKind,
    net::{IpAddr, Ipv4Addr},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

const OBJECT_POINTERS: [&str; 6] = [
    "",
    "/artifact",
    "/execution",
    "/generation",
    "/limits",
    "/http",
];

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        loop {
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "qps-runtime-validation-{}-{sequence}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("Could not create test directory {path:?}: {error}"),
            }
        }
    }

    fn write(&self, contents: &str) -> PathBuf {
        let path = self.0.join("runtime.json");
        fs::write(&path, contents).expect("Could not write test runtime file");
        path
    }

    fn write_json(&self, document: &Value) -> PathBuf {
        self.write(&serde_json::to_string(document).expect("Could not serialize test document"))
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../contracts/fixtures/inference/runtime.json")
}

fn fixture_document() -> Value {
    let contents = fs::read_to_string(fixture_path()).expect("Could not read approved fixture");
    serde_json::from_str(&contents).expect("Approved fixture must contain valid JSON")
}

fn runtime_document(document: &Value) -> Runtime {
    let directory = TestDirectory::new();
    read_runtime(&directory.write_json(document)).expect("Test runtime must deserialize")
}

fn runtime_with(pointer: &str, value: Value) -> Runtime {
    let mut document = fixture_document();
    *document
        .pointer_mut(pointer)
        .expect("Fixture field must exist") = value;
    runtime_document(&document)
}

fn data_error(path: &Path) -> serde_json::Error {
    match read_runtime(path) {
        Err(ContractLoadError::Parse {
            path: error_path,
            source,
        }) => {
            assert_eq!(
                error_path.as_deref(),
                Some(path),
                "Parse errors must retain the input path"
            );
            assert!(source.is_data(), "Expected a data error, got: {source}");
            source
        }
        other => panic!("Expected ContractLoadError::Parse for {path:?}, got: {other:?}"),
    }
}

#[test]
fn approved_fixture_parses_successfully() {
    let runtime = read_runtime(&fixture_path()).expect("Approved runtime fixture should parse");

    assert_eq!(runtime.version, 1);
    assert_eq!(runtime.deployment_id, format!("dep-{}", "a".repeat(32)));
    assert_eq!(
        runtime.model_version_id,
        format!("model-{}", "b".repeat(32))
    );
    assert_eq!(runtime.served_model_name, "qwen-math-r16");
    assert_eq!(runtime.backend, "rust-qwen3");
    assert_eq!(runtime.model_profile, "qwen3-0.6b-v1");
    assert_eq!(runtime.artifact.id, format!("art-{}", "c".repeat(32)));
    assert_eq!(
        runtime.artifact.directory,
        Path::new("/models").join(&runtime.artifact.id)
    );
    assert_eq!(runtime.artifact.manifest_sha, "0".repeat(64));
    assert_eq!(runtime.execution.tensor_parallel_size, 1);
    assert_eq!(runtime.limits.max_context_tokens, 2048);
    assert_eq!(runtime.limits.max_output_tokens, 512);
    assert_eq!(runtime.limits.max_active_generations, 1);
    assert_eq!(runtime.http.bind, IpAddr::V4(Ipv4Addr::LOCALHOST));
    assert_eq!(runtime.http.port, 8001);
}

#[test]
fn missing_file_produces_read_error() {
    let directory = TestDirectory::new();
    let path = directory.0.join("missing.json");

    match read_runtime(&path) {
        Err(ContractLoadError::Read {
            path: error_path,
            source,
        }) => {
            assert_eq!(error_path, path);
            assert_eq!(source.kind(), ErrorKind::NotFound);
        }
        other => panic!("Expected ContractLoadError::Read, got: {other:?}"),
    }
}

#[test]
fn malformed_json_produces_parse_error() {
    let directory = TestDirectory::new();
    let path = directory.write("{ not valid JSON");

    match read_runtime(&path) {
        Err(ContractLoadError::Parse {
            path: error_path,
            source,
        }) => {
            assert_eq!(error_path.as_deref(), Some(path.as_path()));
            assert!(source.is_syntax(), "Expected a syntax error, got: {source}");
            assert_eq!(source.line(), 1);
            assert!(source.column() > 0);
        }
        other => panic!("Expected ContractLoadError::Parse, got: {other:?}"),
    }
}

#[test]
fn missing_required_fields_are_rejected_in_every_object() {
    let directory = TestDirectory::new();
    let fixture = fixture_document();

    for pointer in OBJECT_POINTERS {
        let fields: Vec<String> = fixture
            .pointer(pointer)
            .and_then(Value::as_object)
            .expect("Fixture section must be an object")
            .keys()
            .cloned()
            .collect();

        for field in fields {
            let mut document = fixture.clone();
            document
                .pointer_mut(pointer)
                .and_then(Value::as_object_mut)
                .expect("Fixture section must be an object")
                .remove(&field);

            let error = data_error(&directory.write_json(&document));
            assert!(
                error
                    .to_string()
                    .contains(&format!("missing field `{field}`")),
                "Missing {field:?} at {pointer:?} should be reported, got: {error}"
            );
        }
    }
}

#[test]
fn unknown_fields_are_rejected_in_every_object() {
    let directory = TestDirectory::new();
    let fixture = fixture_document();

    for pointer in OBJECT_POINTERS {
        let mut document = fixture.clone();
        document
            .pointer_mut(pointer)
            .and_then(Value::as_object_mut)
            .expect("Fixture section must be an object")
            .insert("unexpectedField".into(), Value::Bool(true));

        let error = data_error(&directory.write_json(&document));
        assert!(
            error
                .to_string()
                .contains("unknown field `unexpectedField`"),
            "Unknown field at {pointer:?} should be reported, got: {error}"
        );
    }
}

#[test]
fn invalid_enum_values_are_rejected() {
    let directory = TestDirectory::new();
    let fixture = fixture_document();
    let cases = [
        ("/execution/device", "mps"),
        ("/execution/device", "CUDA"),
        ("/execution/dtype", "fp16"),
        ("/execution/dtype", "BF16"),
        ("/generation/mode", "sample"),
        ("/generation/mode", "Greedy"),
    ];

    for (pointer, invalid_value) in cases {
        let mut document = fixture.clone();
        *document
            .pointer_mut(pointer)
            .expect("Enum field must exist") = invalid_value.into();

        let error = data_error(&directory.write_json(&document));
        assert!(
            error
                .to_string()
                .contains(&format!("unknown variant `{invalid_value}`")),
            "Invalid enum at {pointer:?} should be reported, got: {error}"
        );
    }
}

#[test]
fn approved_fixture_satisfies_runtime_value_rules() {
    let runtime = read_runtime(&fixture_path()).expect("Approved fixture should parse");
    validate_runtime(&runtime).expect("Approved fixture should satisfy the runtime contract");
}

#[test]
fn unsupported_versions_report_found_and_expected_versions() {
    for version in [0, 2, u32::MAX] {
        let runtime = runtime_with("/version", version.into());
        match validate_runtime(&runtime) {
            Err(ContractValidationError::UnsupportedVersion { found, expected }) => {
                assert_eq!(found, version);
                assert_eq!(expected, 1);
            }
            other => panic!("Expected UnsupportedVersion for {version}, got: {other:?}"),
        }
    }
}

#[test]
fn invalid_resource_ids_report_the_correct_identity_and_value() {
    let fields = [
        ("/deploymentId", "runtime.deploymentId", "dep-"),
        ("/modelVersionId", "runtime.modelVersionId", "model-"),
        ("/artifact/id", "runtime.artifact.id", "art-"),
    ];

    for (pointer, field, prefix) in fields {
        let invalid_ids = [
            String::new(),
            format!("wrong-{}", "a".repeat(32)),
            prefix.to_owned(),
            format!("{prefix}{}", "a".repeat(31)),
            format!("{prefix}{}", "a".repeat(33)),
            format!("{prefix}{}", "A".repeat(32)),
            format!("{prefix}{}", "g".repeat(32)),
            format!("{prefix}{}-", "a".repeat(31)),
            format!("{prefix}{}é", "a".repeat(30)),
        ];

        for invalid_id in invalid_ids {
            let runtime = runtime_with(pointer, invalid_id.clone().into());
            match validate_runtime(&runtime) {
                Err(ContractValidationError::UnsupportedIdentity {
                    identity,
                    found,
                    expected,
                }) => {
                    assert_eq!(identity, field);
                    assert_eq!(found, invalid_id);
                    assert!(expected.starts_with(prefix));
                    assert!(expected.contains("32"));
                }
                other => panic!(
                    "Expected UnsupportedIdentity for {field}={invalid_id:?}, got: {other:?}"
                ),
            }
        }
    }
}

#[test]
fn resource_ids_and_hash_accept_all_lowercase_hexadecimal_digits() {
    let mut document = fixture_document();
    let suffix = "0123456789abcdef".repeat(2);
    document["deploymentId"] = format!("dep-{suffix}").into();
    document["modelVersionId"] = format!("model-{suffix}").into();
    document["artifact"]["id"] = format!("art-{suffix}").into();
    document["artifact"]["manifestSha"] = "0123456789abcdef".repeat(4).into();

    validate_runtime(&runtime_document(&document))
        .expect("All lowercase hexadecimal digits are valid");
}

#[test]
fn profile_names_accept_valid_characters_and_length_boundaries() {
    for pointer in ["/backend", "/modelProfile"] {
        for value in ["a".to_owned(), "a.0-b".to_owned(), "a".repeat(80)] {
            let runtime = runtime_with(pointer, value.clone().into());
            assert!(
                validate_runtime(&runtime).is_ok(),
                "Valid profile name at {pointer} should pass: {value:?}"
            );
        }
    }
}

#[test]
fn profile_names_reject_invalid_characters_and_lengths() {
    let invalid_names = [
        String::new(),
        "0abc".to_owned(),
        "-abc".to_owned(),
        ".abc".to_owned(),
        "Uppercase".to_owned(),
        "a_b".to_owned(),
        "a/b".to_owned(),
        "a b".to_owned(),
        "aé".to_owned(),
        "a".repeat(81),
    ];

    for (pointer, field) in [
        ("/backend", "runtime.backend"),
        ("/modelProfile", "runtime.modelProfile"),
    ] {
        for value in &invalid_names {
            let runtime = runtime_with(pointer, value.clone().into());
            match validate_runtime(&runtime) {
                Err(ContractValidationError::UnsupportedProfile {
                    profile,
                    found,
                    expected,
                }) => {
                    assert_eq!(profile, field);
                    assert_eq!(&found, value);
                    assert!(!expected.is_empty());
                }
                other => {
                    panic!("Expected UnsupportedProfile for {field}={value:?}, got: {other:?}")
                }
            }
        }
    }
}

#[test]
fn served_model_names_accept_valid_characters_and_length_boundaries() {
    for name in [
        "a".to_owned(),
        "0".to_owned(),
        "A_b.0-c".to_owned(),
        "a".repeat(80),
    ] {
        let runtime = runtime_with("/servedModelName", name.clone().into());
        assert!(
            validate_runtime(&runtime).is_ok(),
            "Valid served model name should pass: {name:?}"
        );
    }
}

#[test]
fn served_model_names_reject_invalid_characters_and_lengths() {
    for name in [
        String::new(),
        "_model".to_owned(),
        "-model".to_owned(),
        ".model".to_owned(),
        "a/b".to_owned(),
        "a b".to_owned(),
        "é".to_owned(),
        "a💡".to_owned(),
        "a".repeat(81),
    ] {
        let runtime = runtime_with("/servedModelName", name.clone().into());
        match validate_runtime(&runtime) {
            Err(ContractValidationError::UnsupportedServedModelName { found, expected }) => {
                assert_eq!(found, name);
                assert!(!expected.is_empty());
            }
            other => panic!("Expected UnsupportedServedModelName for {name:?}, got: {other:?}"),
        }
    }
}

#[test]
fn manifest_hash_rejects_wrong_lengths_uppercase_and_non_hexadecimal_bytes() {
    for value in [
        String::new(),
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
        " ".repeat(64),
        "💡".repeat(16),
    ] {
        let runtime = runtime_with("/artifact/manifestSha", value.clone().into());
        match validate_runtime(&runtime) {
            Err(ContractValidationError::UnsupportedHash {
                hash,
                found,
                expected,
            }) => {
                assert_eq!(hash, "runtime.artifact.manifestSha");
                assert_eq!(found, value);
                assert!(expected.contains("64"));
            }
            other => panic!("Expected UnsupportedHash for {value:?}, got: {other:?}"),
        }
    }
}

#[test]
fn relative_artifact_directories_are_rejected() {
    for value in [
        "",
        "models/art",
        "./models",
        "../models",
        "~/models",
        "s3://bucket/model",
    ] {
        let runtime = runtime_with("/artifact/directory", value.into());
        match validate_runtime(&runtime) {
            Err(ContractValidationError::InvalidArtifactDirectory { found, expected }) => {
                assert_eq!(found, Path::new(value));
                assert!(!expected.is_empty());
            }
            other => panic!("Expected InvalidArtifactDirectory for {value:?}, got: {other:?}"),
        }
    }
}

#[test]
fn absolute_artifact_paths_pass_without_requiring_artifact_files() {
    for value in ["/", "/models/art", "/models/qps-fixture-not-downloaded"] {
        let runtime = runtime_with("/artifact/directory", value.into());
        validate_runtime(&runtime).expect("Artifact existence is checked in a later stage");
    }
}

#[test]
fn numeric_rules_accept_inclusive_boundaries() {
    let cases: &[(&str, &[u32])] = &[
        ("/execution/tensorParallelSize", &[1, 2]),
        ("/limits/maxContextTokens", &[16, 2048]),
        ("/limits/maxOutputTokens", &[1, 512]),
        ("/limits/maxActiveGenerations", &[1, u32::MAX]),
        ("/http/port", &[1, u32::from(u16::MAX)]),
    ];

    for &(pointer, values) in cases {
        for &value in values {
            let runtime = runtime_with(pointer, value.into());
            assert!(
                validate_runtime(&runtime).is_ok(),
                "Valid boundary at {pointer} should pass: {value}"
            );
        }
    }
}

#[test]
fn invalid_numeric_values_report_the_correct_field_and_value() {
    let cases: &[(&str, &str, &[u32])] = &[
        (
            "/execution/tensorParallelSize",
            "runtime.execution.tensorParallelSize",
            &[0, 3, u32::MAX],
        ),
        (
            "/limits/maxContextTokens",
            "runtime.limits.maxContextTokens",
            &[0, 15, 2049, u32::MAX],
        ),
        (
            "/limits/maxOutputTokens",
            "runtime.limits.maxOutputTokens",
            &[0, 513, u32::MAX],
        ),
        (
            "/limits/maxActiveGenerations",
            "runtime.limits.maxActiveGenerations",
            &[0],
        ),
        ("/http/port", "runtime.http.port", &[0]),
    ];

    for &(pointer, field, values) in cases {
        for &invalid_value in values {
            let runtime = runtime_with(pointer, invalid_value.into());
            match validate_runtime(&runtime) {
                Err(ContractValidationError::InvalidNumericValue {
                    value,
                    found,
                    expected,
                }) => {
                    assert_eq!(value, field);
                    assert_eq!(found, invalid_value);
                    assert!(!expected.is_empty());
                }
                other => panic!(
                    "Expected InvalidNumericValue for {field}={invalid_value}, got: {other:?}"
                ),
            }
        }
    }
}

#[test]
fn ports_outside_u16_range_are_rejected_during_deserialization() {
    let directory = TestDirectory::new();
    for value in [-1_i64, 65_536] {
        let mut document = fixture_document();
        document["http"]["port"] = value.into();
        data_error(&directory.write_json(&document));
    }
}

#[test]
fn bind_address_requires_exactly_ipv4_localhost() {
    for address in ["0.0.0.0", "127.0.0.2", "192.0.2.1", "::1", "::"] {
        let runtime = runtime_with("/http/bind", address.into());
        match validate_runtime(&runtime) {
            Err(ContractValidationError::InvalidBindAddress { found, expected }) => {
                assert_eq!(
                    found,
                    address.parse::<IpAddr>().expect("Test address must parse")
                );
                assert_eq!(expected, IpAddr::V4(Ipv4Addr::LOCALHOST));
            }
            other => panic!("Expected InvalidBindAddress for {address:?}, got: {other:?}"),
        }
    }
}

#[test]
fn runtime_value_checks_do_not_enforce_backend_execution_capabilities() {
    let mut document = fixture_document();
    document["execution"]["device"] = "cpu".into();
    document["execution"]["dtype"] = "fp32".into();
    document["execution"]["tensorParallelSize"] = 2.into();
    document["limits"]["maxActiveGenerations"] = 2.into();

    validate_runtime(&runtime_document(&document))
        .expect("Backend support is checked separately from runtime document values");
}

#[test]
fn cli_prints_the_runtime_after_successful_validation() {
    let output = Command::new(env!("CARGO_BIN_EXE_qps-inference"))
        .arg("--runtime")
        .arg(fixture_path())
        .output()
        .expect("Could not run qps-inference");

    assert!(output.status.success(), "CLI failed: {output:?}");
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).expect("Fixture output should be UTF-8");
    assert!(
        stdout.contains("Runtime {"),
        "Expected the Runtime, got: {stdout}"
    );
    assert!(stdout.contains("qwen-math-r16"));
}

#[test]
fn cli_returns_failure_and_no_success_output_for_invalid_runtime_values() {
    let directory = TestDirectory::new();
    for (pointer, value, error_variant) in [
        ("/version", Value::from(2), "UnsupportedVersion"),
        ("/http/port", Value::from(0), "InvalidNumericValue"),
    ] {
        let mut document = fixture_document();
        *document
            .pointer_mut(pointer)
            .expect("Fixture field must exist") = value;
        let path = directory.write_json(&document);
        let output = Command::new(env!("CARGO_BIN_EXE_qps-inference"))
            .arg("--runtime")
            .arg(path)
            .output()
            .expect("Could not run qps-inference");

        assert_eq!(
            output.status.code(),
            Some(1),
            "CLI accepted invalid {pointer}"
        );
        assert!(
            output.stdout.is_empty(),
            "Failure must not print success output"
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains(error_variant));
    }
}

#[test]
fn cli_argument_errors_return_usage_status_without_success_output() {
    let fixture = fixture_path().into_os_string();
    let cases = [
        vec![],
        vec!["--wrong".into(), fixture.clone()],
        vec!["--runtime".into()],
        vec!["--runtime".into(), fixture, "--extra".into()],
    ];

    for args in cases {
        let output = Command::new(env!("CARGO_BIN_EXE_qps-inference"))
            .args(&args)
            .output()
            .expect("Could not run qps-inference");

        assert_eq!(
            output.status.code(),
            Some(2),
            "Wrong exit status for {args:?}"
        );
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains("Usage:"));
    }
}

#[test]
fn cli_read_and_parse_errors_return_failure_status() {
    let directory = TestDirectory::new();
    let malformed = directory.write("{ not valid JSON");
    let missing = directory.0.join("missing.json");

    for (path, error_variant) in [(missing, "Read {"), (malformed, "Parse {")] {
        let output = Command::new(env!("CARGO_BIN_EXE_qps-inference"))
            .arg("--runtime")
            .arg(path)
            .output()
            .expect("Could not run qps-inference");

        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(String::from_utf8_lossy(&output.stderr).contains(error_variant));
    }
}
