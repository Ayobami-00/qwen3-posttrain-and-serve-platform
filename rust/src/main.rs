use std::{eprintln, path::PathBuf, println};

use qps_inference::{read_runtime, validate_runtime};

fn main() {
    let mut args = std::env::args_os();

    let _program = args.next();
    let flag = args.next();
    let runtime_path = args.next();

    let usage_print = || {
        eprintln!("Usage: qps-inference --runtime <path>");
    };

    let Some(parsed_flag) = flag else {
        eprintln!("Missing required --runtime argument.");
        usage_print();
        std::process::exit(2);
    };

    if parsed_flag != "--runtime" {
        eprintln!("Wrong required argument passed.");
        usage_print();
        std::process::exit(2);
    }

    let Some(parsed_runtime_path) = runtime_path else {
        eprintln!("Missing required runtime path.");
        usage_print();
        std::process::exit(2);
    };

    if args.next().is_some() {
        eprintln!("Only the runtime argument is required");
        usage_print();
        std::process::exit(2);
    }

    let parsed_runtime_path_buf = PathBuf::from(parsed_runtime_path);

    match read_runtime(&parsed_runtime_path_buf) {
        Ok(runtime) => match validate_runtime(&runtime) {
            Ok(_) => {
                println!("{runtime:#?}")
            }
            Err(error) => {
                eprintln!("Failed to validate runtime:: {error:?}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("Failed to read runtime:: {error:?}");
            std::process::exit(1);
        }
    }
}
