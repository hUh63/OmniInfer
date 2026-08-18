use super::*;

#[test]
fn format_model_menu_renders_core_columns() {
    let items = [ModelMenuItem {
        label: "qwen/Qwen3.5-4B-Q4_K_M.gguf".to_string(),
        provider: "qwen".to_string(),
        quant: "Q4_K_M".to_string(),
        disk: "2.31 GiB".to_string(),
        ctx: "32k".to_string(),
        fit: "good".to_string(),
        backend: "llama.cpp-linux-cuda".to_string(),
        evidence: "direct/high".to_string(),
        selected: true,
    }];
    let table = format_model_menu_for_width(&items, None, "", 140);
    assert!(table.contains("Model"));
    assert!(table.contains("Provider"));
    assert!(table.contains("Q4_K_M"));
    assert!(table.contains("32k"));
    assert!(table.contains("llama.cpp"));
    assert!(table.contains("direct"));
    assert!(table.contains("*"));
}

#[test]
fn model_menu_screen_renders_context_panels() {
    let context = ModelMenuContext {
        hardware_lines: vec![
            "Host: linux x86_64 | CPU: 24 threads | RAM: 80.0 GiB free / 125.0 GiB total"
                .to_string(),
            "GPU: NVIDIA RTX 3090 x8 | best free GPU 0: 23.0 GiB free / 24.0 GiB total".to_string(),
        ],
        backend_line: "Backend: llama.cpp-linux-cuda (installed, compatible)".to_string(),
    };
    let items = [ModelMenuItem {
        label: "qwen/model.gguf".to_string(),
        provider: "qwen".to_string(),
        quant: "Q4_K_M".to_string(),
        disk: "2 GiB".to_string(),
        ctx: "32k".to_string(),
        fit: "good".to_string(),
        backend: "llama.cpp-linux-cuda".to_string(),
        evidence: "direct/high".to_string(),
        selected: true,
    }];
    let screen = format_model_menu_screen_for_width(
        "OmniInfer",
        "Local model picker",
        &context,
        &items,
        Some(0),
        "",
        120,
    );
    assert!(screen.contains("Host: linux"));
    assert!(screen.contains("Backend: llama.cpp-linux-cuda"));
    assert!(screen.contains("Provider"));
    assert!(screen.contains('┌'));
    assert!(screen.contains('└'));
    assert!(!screen.contains("+--"));
}

#[test]
fn format_model_menu_marks_cursor_and_prompt_buffer() {
    let items = [
        ModelMenuItem {
            label: "first.gguf".to_string(),
            provider: "local".to_string(),
            quant: "Q4_K_M".to_string(),
            disk: "2 GiB".to_string(),
            ctx: "32k".to_string(),
            fit: "good".to_string(),
            backend: "llama.cpp-linux-cuda".to_string(),
            evidence: "direct/high".to_string(),
            selected: false,
        },
        ModelMenuItem {
            label: "second.gguf".to_string(),
            provider: "local".to_string(),
            quant: "Q8_0".to_string(),
            disk: "4 GiB".to_string(),
            ctx: "128k".to_string(),
            fit: "good".to_string(),
            backend: "llama.cpp-linux-cuda".to_string(),
            evidence: "direct/high".to_string(),
            selected: true,
        },
    ];
    let table = format_model_menu_for_width(&items, Some(1), "2", 120);
    assert!(table.contains(">    2  *"));
    assert!(table.contains("Number: 2"));
    assert!(table.contains("Up/Down"));
}

#[test]
fn format_model_menu_hides_empty_number_prompt() {
    let items = [ModelMenuItem {
        label: "first.gguf".to_string(),
        provider: "local".to_string(),
        quant: "Q4_K_M".to_string(),
        disk: "2 GiB".to_string(),
        ctx: "32k".to_string(),
        fit: "good".to_string(),
        backend: "llama.cpp-linux-cuda".to_string(),
        evidence: "direct/high".to_string(),
        selected: false,
    }];
    let table = format_model_menu_for_width(&items, Some(0), "", 120);
    assert!(!table.contains("Select:"));
    assert!(!table.contains("Number:"));
}

#[test]
fn buffered_model_index_uses_one_based_numbers() {
    assert_eq!(buffered_model_index("1", 3), Some(0));
    assert_eq!(buffered_model_index("3", 3), Some(2));
    assert_eq!(buffered_model_index("0", 3), None);
    assert_eq!(buffered_model_index("4", 3), None);
}

#[test]
fn model_menu_columns_fit_narrow_terminals() {
    let narrow = model_menu_columns(34);
    assert!(narrow.optional.is_empty());
    assert!(narrow.model >= 8);

    let medium = model_menu_columns(76);
    assert!(
        medium
            .optional
            .iter()
            .any(|column| column.kind == ModelMenuColumnKind::Fit)
    );
    assert!(
        !medium
            .optional
            .iter()
            .any(|column| column.kind == ModelMenuColumnKind::Backend)
    );

    let wide = model_menu_columns(160);
    assert_eq!(wide.optional.len(), MENU_COLUMNS_FULL.len());
    let row = model_menu_row("", "1", "", "model", &wide.header_values(), &wide);
    assert!(row.len() <= 160);
}

#[test]
fn truncate_cell_marks_long_values() {
    assert_eq!(truncate_cell("abcdefg", 6), "abc...");
    assert_eq!(truncate_cell("abc", 4), "abc");
    assert_eq!(truncate_cell_plain("abcdefg", 6), "abcdef");
}
