from pathlib import Path

try:
    from src.utils.update_readme_utils import generate_md_links
except ModuleNotFoundError:
    from update_readme_utils import generate_md_links


def make_hyperlink(class_name, script_path):
    return f"[{class_name}](./{script_path}#{class_name.lower()})"


def generate_markdown_table_from_folders(*folders):
    """Generates a markdown table using links to classes from Python scripts in the provided folders."""
    headers, all_hyperlinks = [], []
    for folder in folders:
        headers.append(folder.name)  # use folder name as header

        links = generate_md_links(folder)
        all_hyperlinks.append(links)

    # Create table header
    header = "| " + " | ".join(f"**{h}**" for h in headers) + " |"
    separator = "| " + " | ".join("-" * len(h) for h in headers) + " |"

    # Handle uneven lengths among the lists
    max_len = max(len(hyperlinks) for hyperlinks in all_hyperlinks)

    rows = []
    for i in range(max_len):
        row_links = [
            hyperlinks[i] if i < len(hyperlinks) else ""
            for hyperlinks in all_hyperlinks
        ]
        row = "| " + " | ".join(row_links) + " |"
        rows.append(row)

    # Combine everything into a single markdown table string
    table = "\n".join([header, separator] + rows)
    return table


def write_table():
    seed_table = Path("public", "README_SEED.MD")
    seed_content = seed_table.read_text()

    # -------- table 1 --------
    table = generate_markdown_table_from_folders(
        Path("src", "modules", "logits"),
        Path("src", "modules", "masks"),
        Path("src", "modules", "joint"),
    )
    content = seed_content.replace("<!-- TABLE -->", table)

    # -------- table 2 --------
    table = generate_markdown_table_from_folders(
        Path("src", "analysis"),
        Path("src", "analysis", "run_loader_image"),
        Path("src", "analysis", "run_loader_results"),
        Path("src", "analysis", "run_loader_manual_inspect"),
    )
    content = content.replace("<!-- TABLE Tests -->", table)

    Path("README.MD").write_text(content)


if __name__ == "__main__":
    write_table()
