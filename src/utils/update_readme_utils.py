"""
functions that converts a folder of python scripts into markdown hyperlinks

- searchs for all classes in that folder
- produces a markdown hyperlink for each class

usage:
    path_obj = Path("src", "modules", "logits")
    links = generate_md_links(path_obj)
    for link in links:
        print(link)

"""

import ast
from pathlib import Path


def extract_classes_and_functions_from_script(script_path):
    """
    Extracts class and function names from a given Python script, excluding class methods.
    """
    with open(script_path, "r") as file:
        parsed_ast = ast.parse(file.read())

    classes = [
        node.name for node in ast.walk(parsed_ast) if isinstance(node, ast.ClassDef)
    ]

    # Extracting top-level functions only (i.e., those not nested inside classes or other functions)
    functions = [
        node.name
        for node in ast.iter_child_nodes(parsed_ast)
        if isinstance(node, ast.FunctionDef)
    ]

    return classes, functions


def generate_md_links(folder_path):
    """
    Generates markdown hyperlinks for classes and top-level functions in python scripts from the given folder.
    """
    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} is not a directory!")

    md_links = []
    all_scripts_paths = folder_path.glob("*.py")
    all_scripts_paths = [
        path for path in all_scripts_paths if not path.name.startswith("_")
    ]
    for script_path in all_scripts_paths:
        classes, functions = extract_classes_and_functions_from_script(script_path)
        for name in classes + functions:
            link = f"[{name}](./{script_path.as_posix()}#{name.lower()})"
            md_links.append(link)
    return md_links


if __name__ == "__main__":
    path_obj = Path("src", "modules", "logits")
    links = generate_md_links(path_obj)
    for link in links:
        print(link)
