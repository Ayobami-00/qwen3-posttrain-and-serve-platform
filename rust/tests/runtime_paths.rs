use qps_inference::{ContractLoadError, read_runtime};
use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

const RUNTIME_FIXTURE: &str = include_str!("../../contracts/fixtures/inference/runtime.json");
static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        loop {
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let directory = std::env::temp_dir().join(format!(
                "qps-runtime-paths-{}-{sequence}",
                std::process::id()
            ));
            match fs::create_dir(&directory) {
                Ok(()) => return Self(directory),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("Could not create test directory: {error}"),
            }
        }
    }

    #[cfg(unix)]
    fn non_unicode_path(&self) -> PathBuf {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let path = self
            .0
            .join(OsString::from_vec(b"runtime-\xff.json".to_vec()));
        assert!(path.to_str().is_none(), "Test path must be non-Unicode");
        path
    }

    #[cfg(target_os = "linux")]
    fn non_unicode_runtime(&self) -> PathBuf {
        let path = self.non_unicode_path();
        fs::write(&path, RUNTIME_FIXTURE).unwrap();
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn assert_cli_success(path: &Path) {
    let output = Command::new(env!("CARGO_BIN_EXE_qps-inference"))
        .arg("--runtime")
        .arg(path)
        .output()
        .expect("Could not run qps-inference");

    assert!(output.status.success(), "CLI failed: {output:?}");
    assert!(
        output.stderr.is_empty(),
        "Unexpected diagnostic: {output:?}"
    );
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("qwen-math-r16")
    );
}

#[test]
fn non_utf8_document_contents_preserve_the_io_error_and_path() {
    let directory = TestDirectory::new();
    let path = directory.0.join("runtime.json");
    fs::write(&path, [0xff, 0xfe]).unwrap();

    match read_runtime(&path) {
        Err(ContractLoadError::Read {
            path: reported,
            source,
        }) => {
            assert_eq!(reported, path);
            assert_eq!(source.kind(), ErrorKind::InvalidData);
        }
        other => panic!("Invalid UTF-8 contents must be a read error, got: {other:?}"),
    }
}

#[test]
fn cli_accepts_spaces_and_unicode_in_the_document_path() {
    let directory = TestDirectory::new();
    let folder = directory.0.join("runtime fixtures 配置");
    fs::create_dir(&folder).unwrap();
    let path = folder.join("valid runtime.json");
    fs::write(&path, RUNTIME_FIXTURE).unwrap();

    assert_cli_success(&path);
}

#[cfg(unix)]
#[test]
fn missing_non_unicode_path_preserves_the_read_error_path() {
    let directory = TestDirectory::new();
    let path = directory.non_unicode_path();

    match read_runtime(&path) {
        Err(ContractLoadError::Read { path: reported, .. }) => assert_eq!(reported, path),
        other => panic!("An unreadable OS-native path must return its read error, got: {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn cli_reports_a_read_failure_for_a_missing_non_unicode_path() {
    let directory = TestDirectory::new();
    let output = Command::new(env!("CARGO_BIN_EXE_qps-inference"))
        .arg("--runtime")
        .arg(directory.non_unicode_path())
        .output()
        .expect("OS-native arguments must reach the executable");

    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Failed to read runtime"));
}

// The macOS filesystem rejects these filename bytes; Linux CI covers existing files.
#[cfg(target_os = "linux")]
#[test]
fn public_reader_accepts_a_non_unicode_document_path() {
    let directory = TestDirectory::new();
    let path = directory.non_unicode_runtime();
    let runtime = read_runtime(&path).expect("OS-native paths must be readable");

    assert_eq!(runtime.served_model_name, "qwen-math-r16");
    qps_inference::validate_runtime(&runtime).expect("Fixture values must remain valid");
}

#[cfg(target_os = "linux")]
#[test]
fn cli_preserves_a_non_unicode_document_path() {
    let directory = TestDirectory::new();
    assert_cli_success(&directory.non_unicode_runtime());
}
