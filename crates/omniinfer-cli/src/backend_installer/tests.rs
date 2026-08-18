use super::*;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::io;

enum TestTarEntry<'a> {
    File(&'a str, &'a [u8]),
    Symlink(&'a str, &'a str),
    HardLink(&'a str, &'a str),
    Special(&'a str),
    RawFilePath(&'a str),
}

#[test]
fn tar_links_extract_and_survive_runtime_staging() {
    let archive = build_test_tar(&[
        TestTarEntry::File("runtime/llama-server", b"launcher"),
        TestTarEntry::File("runtime/libreal.dylib", b"library"),
        TestTarEntry::Symlink("runtime/libalias.dylib", "libreal.dylib"),
        TestTarEntry::HardLink("runtime/libhard.dylib", "runtime/libreal.dylib"),
    ]);
    let extracted = test_dir("safe-links-extracted");
    extract_archive(&archive, "tar.gz", &extracted).expect("extract safe links");

    let symlink = extracted.join("runtime/libalias.dylib");
    assert!(
        fs::symlink_metadata(&symlink)
            .expect("symlink metadata")
            .file_type()
            .is_symlink()
    );
    assert_eq!(fs::read_link(&symlink).unwrap(), Path::new("libreal.dylib"));
    assert_eq!(
        fs::read(extracted.join("runtime/libhard.dylib")).unwrap(),
        b"library"
    );

    let staged = test_dir("safe-links-staged");
    copy_dir_recursive(&extracted.join("runtime"), &staged).expect("stage runtime links");
    let staged_symlink = staged.join("libalias.dylib");
    assert!(
        fs::symlink_metadata(&staged_symlink)
            .expect("staged symlink metadata")
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        fs::canonicalize(&staged_symlink).unwrap(),
        fs::canonicalize(staged.join("libreal.dylib")).unwrap()
    );

    fs::remove_dir_all(extracted).ok();
    fs::remove_dir_all(staged).ok();
}

#[test]
fn tar_extractor_rejects_unsafe_paths_links_and_entry_types() {
    let cases = [
        (
            "absolute-path",
            build_test_tar(&[TestTarEntry::RawFilePath("/tmp/escape")]),
            "unsafe archive path",
        ),
        (
            "parent-path",
            build_test_tar(&[TestTarEntry::RawFilePath("../escape")]),
            "unsafe archive path",
        ),
        (
            "absolute-link",
            build_test_tar(&[
                TestTarEntry::File("runtime/real", b"safe"),
                TestTarEntry::Symlink("runtime/link", "/tmp/escape"),
            ]),
            "tar link target must be relative",
        ),
        (
            "escaping-link",
            build_test_tar(&[
                TestTarEntry::File("runtime/real", b"safe"),
                TestTarEntry::Symlink("runtime/link", "../../escape"),
            ]),
            "tar link target escapes staging root",
        ),
        (
            "dangling-link",
            build_test_tar(&[TestTarEntry::Symlink("runtime/link", "missing")]),
            "tar link target does not exist",
        ),
        (
            "link-cycle",
            build_test_tar(&[
                TestTarEntry::Symlink("runtime/one", "two"),
                TestTarEntry::Symlink("runtime/two", "one"),
            ]),
            "tar symbolic link cycle",
        ),
        (
            "escaping-hard-link",
            build_test_tar(&[TestTarEntry::HardLink("runtime/link", "../escape")]),
            "tar link target escapes staging root",
        ),
        (
            "special-entry",
            build_test_tar(&[TestTarEntry::Special("runtime/device")]),
            "unsupported tar entry type",
        ),
    ];

    for (name, archive, expected) in cases {
        let destination = test_dir(name);
        let error =
            extract_archive(&archive, "tar.gz", &destination).expect_err("unsafe tar must fail");
        assert!(
            error.to_string().contains(expected),
            "{name}: expected {expected:?}, got {error:#}"
        );
        fs::remove_dir_all(destination).ok();
    }
}

#[test]
fn tar_extractor_rejects_canonical_target_outside_staging() {
    let destination = test_dir("canonical-target-destination");
    let outside = test_dir("canonical-target-outside");
    fs::write(outside.join("library.dylib"), "outside").unwrap();
    fs::create_dir_all(destination.join("runtime")).unwrap();
    std::os::unix::fs::symlink(
        outside.join("library.dylib"),
        destination.join("runtime/external"),
    )
    .unwrap();
    let archive = build_test_tar(&[TestTarEntry::Symlink("runtime/link.dylib", "external")]);

    let error = extract_archive(&archive, "tar.gz", &destination)
        .expect_err("canonical target outside staging must fail");
    assert!(
        error
            .to_string()
            .contains("tar link target escapes staging root")
    );

    fs::remove_dir_all(destination).ok();
    fs::remove_dir_all(outside).ok();
}

#[test]
fn runtime_staging_rejects_symbolic_links_outside_source_root() {
    let source = test_dir("copy-external-source");
    let destination = test_dir("copy-external-destination");
    let outside = test_dir("copy-external-outside");
    fs::write(outside.join("library.dylib"), "outside").unwrap();
    let outside_name = outside.file_name().expect("outside directory name");
    std::os::unix::fs::symlink(
        Path::new("..").join(outside_name).join("library.dylib"),
        source.join("external.dylib"),
    )
    .unwrap();

    let error = copy_dir_recursive(&source, &destination)
        .expect_err("runtime staging must reject external symbolic link");
    assert!(
        format!("{error:#}").contains("runtime symbolic link escapes source root"),
        "{error:#}"
    );

    fs::remove_dir_all(source).ok();
    fs::remove_dir_all(destination).ok();
    fs::remove_dir_all(outside).ok();
}

fn test_dir(name: &str) -> PathBuf {
    let path = temp_install_dir(name).expect("test temp path");
    fs::remove_dir_all(&path).ok();
    fs::create_dir_all(&path).expect("create test directory");
    path
}

fn build_test_tar(entries: &[TestTarEntry<'_>]) -> Vec<u8> {
    let encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut builder = tar::Builder::new(encoder);
    for entry in entries {
        match entry {
            TestTarEntry::File(path, contents) => {
                let mut header = tar::Header::new_gnu();
                header.set_path(path).unwrap();
                header.set_size(contents.len() as u64);
                header.set_mode(0o755);
                header.set_cksum();
                builder.append(&header, *contents).unwrap();
            }
            TestTarEntry::Symlink(path, target) => {
                append_test_link(&mut builder, path, target, tar::EntryType::Symlink);
            }
            TestTarEntry::HardLink(path, target) => {
                append_test_link(&mut builder, path, target, tar::EntryType::Link);
            }
            TestTarEntry::Special(path) => {
                let mut header = tar::Header::new_gnu();
                header.set_path(path).unwrap();
                header.set_size(0);
                header.set_mode(0o600);
                header.set_entry_type(tar::EntryType::Char);
                header.set_cksum();
                builder.append(&header, io::empty()).unwrap();
            }
            TestTarEntry::RawFilePath(path) => {
                let mut header = tar::Header::new_gnu();
                header.set_size(0);
                header.set_mode(0o600);
                set_raw_header_value(&mut header, 0, 100, path.as_bytes());
                header.set_cksum();
                builder.append(&header, io::empty()).unwrap();
            }
        }
    }
    let encoder = builder.into_inner().expect("finish tar");
    encoder.finish().expect("finish gzip")
}

fn append_test_link(
    builder: &mut tar::Builder<GzEncoder<Vec<u8>>>,
    path: &str,
    target: &str,
    entry_type: tar::EntryType,
) {
    let mut header = tar::Header::new_gnu();
    header.set_path(path).unwrap();
    header.set_link_name(target).unwrap();
    header.set_size(0);
    header.set_mode(0o777);
    header.set_entry_type(entry_type);
    header.set_cksum();
    builder.append(&header, io::empty()).unwrap();
}

fn set_raw_header_value(header: &mut tar::Header, offset: usize, size: usize, value: &[u8]) {
    assert!(value.len() < size);
    let bytes = header.as_mut_bytes();
    bytes[offset..offset + size].fill(0);
    bytes[offset..offset + value.len()].copy_from_slice(value);
}
