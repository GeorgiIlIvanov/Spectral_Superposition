from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    folder_path="start",
    repo_id="Theopha/Spectral_Superposition",
    repo_type="dataset",
)
print("Upload complete!")
