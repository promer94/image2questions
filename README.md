# JSON to Word Tables

This project creates (and extends) a Word document consisting of borderless `3×2` tables (first row merged for the title and the remaining cells for the four options labeled A→D); each title row uses Word's numbered list style so the questions stay sequential, and any numbering already present in the JSON title is dropped before applying Word's numbering.

## Setup (with `uv`)

1. Initialize or switch to the project environment (replace `myenv` if you prefer another name):

   ```powershell
   uv env new myenv
   uv env use myenv
   ```

2. Add the required dependency with `uv`:

   ```powershell
   uv add python-docx
   ```

3. Install dependencies (the `uv` cache will ensure the lockfile stays updated):

   ```powershell
   uv install
   ```

## Usage

```powershell
uv run python json_to_word.py sample_data.json questions.docx
```

The script expects a JSON array of `Item` objects (`title` + `options` with lowercase `a`-`d`). It writes a Word document where each table matches the described layout and all borders are disabled.

Custom JSON files can be passed by replacing `sample_data.json`, and the output path can be changed as needed.