"""
Tests for utility modules.

This module tests image and file utility functions.
"""

import json
from pathlib import Path

import pytest

from src.utils.image_utils import (
    encode_image_to_base64,
    get_image_mime_type,
    is_valid_image,
    get_image_paths_from_directory,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.utils.file_utils import (
    ensure_directory,
    read_json_file,
    write_json_file,
    get_unique_filename,
    is_valid_path,
)


class TestImageUtils:
    """Tests for image utility functions."""
    
    def test_supported_extensions(self):
        """Test that common image extensions are supported."""
        assert ".jpg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".jpeg" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".png" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".gif" in SUPPORTED_IMAGE_EXTENSIONS
        assert ".webp" in SUPPORTED_IMAGE_EXTENSIONS
    
    def test_get_image_mime_type(self):
        """Test MIME type detection."""
        assert get_image_mime_type("test.jpg") == "image/jpeg"
        assert get_image_mime_type("test.jpeg") == "image/jpeg"
        assert get_image_mime_type("test.png") == "image/png"
        assert get_image_mime_type("test.gif") == "image/gif"
        assert get_image_mime_type("test.webp") == "image/webp"
    
    def test_get_mime_type_case_insensitive(self):
        """Test that MIME type detection is case insensitive."""
        assert get_image_mime_type("test.JPG") == "image/jpeg"
        assert get_image_mime_type("test.PNG") == "image/png"
    
    def test_unknown_extension_defaults_to_jpeg(self):
        """Test that unknown extensions default to image/jpeg."""
        assert get_image_mime_type("test.unknown") == "image/jpeg"
    
    def test_is_valid_image_with_valid_file(self, create_test_image):
        """Test validating a real image file."""
        image_path = create_test_image("valid.jpg")
        assert is_valid_image(image_path) is True
    
    def test_is_valid_image_nonexistent(self, temp_dir):
        """Test that nonexistent files are invalid."""
        assert is_valid_image(temp_dir / "nonexistent.jpg") is False
    
    def test_is_valid_image_wrong_extension(self, temp_dir):
        """Test that non-image extensions are invalid."""
        text_file = temp_dir / "test.txt"
        text_file.write_text("not an image")
        assert is_valid_image(text_file) is False
    
    def test_encode_image_to_base64(self, create_test_image):
        """Test encoding an image to base64."""
        image_path = create_test_image("test.jpg")
        result = encode_image_to_base64(image_path)
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Base64 should only contain valid characters
        import base64
        try:
            base64.b64decode(result)
        except Exception:
            pytest.fail("Result is not valid base64")
    
    def test_encode_nonexistent_image(self, temp_dir):
        """Test encoding a nonexistent image raises error."""
        with pytest.raises(FileNotFoundError):
            encode_image_to_base64(temp_dir / "nonexistent.jpg")
    
    def test_get_image_paths_from_directory(self, temp_dir, create_test_image):
        """Test getting image paths from a directory."""
        # Create some test images
        create_test_image("image1.jpg")
        create_test_image("image2.png")
        (temp_dir / "not_image.txt").write_text("text")
        
        paths = get_image_paths_from_directory(temp_dir)
        
        assert len(paths) == 2
        assert all(p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS for p in paths)
    
    def test_get_image_paths_nonexistent_directory(self, temp_dir):
        """Test getting paths from nonexistent directory raises error."""
        with pytest.raises(FileNotFoundError):
            get_image_paths_from_directory(temp_dir / "nonexistent")


class TestFileUtils:
    """Tests for file utility functions."""
    
    def test_ensure_directory_creates_new(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new" / "nested" / "directory"
        result = ensure_directory(new_dir)
        
        assert result.exists()
        assert result.is_dir()
    
    def test_ensure_directory_existing(self, temp_dir):
        """Test with an existing directory."""
        result = ensure_directory(temp_dir)
        
        assert result.exists()
        assert result == temp_dir
    
    def test_read_json_file(self, sample_json_file):
        """Test reading a JSON file."""
        data = read_json_file(sample_json_file)
        
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["title"] == "What is the capital of France?"
    
    def test_read_json_file_nonexistent(self, temp_dir):
        """Test reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_json_file(temp_dir / "nonexistent.json")
    
    def test_write_json_file(self, temp_dir):
        """Test writing a JSON file."""
        data = {"key": "value", "list": [1, 2, 3]}
        output_path = temp_dir / "output.json"
        
        result = write_json_file(output_path, data)
        
        assert result == output_path
        assert output_path.exists()
        
        # Verify content
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_write_json_file_with_unicode(self, temp_dir):
        """Test writing JSON with unicode characters."""
        data = {"题目": "中文内容", "答案": "选项A"}
        output_path = temp_dir / "unicode.json"
        
        write_json_file(output_path, data, ensure_ascii=False)
        
        content = output_path.read_text(encoding="utf-8")
        assert "题目" in content
        assert "中文内容" in content
    
    def test_write_json_file_append_mode(self, temp_dir):
        """Test appending to existing JSON file."""
        output_path = temp_dir / "append.json"
        
        # Write initial data
        write_json_file(output_path, [1, 2, 3])
        
        # Append more data
        write_json_file(output_path, [4, 5, 6], append=True)
        
        # Verify combined content
        data = read_json_file(output_path)
        assert data == [1, 2, 3, 4, 5, 6]
    
    def test_get_unique_filename_new_file(self, temp_dir):
        """Test getting unique filename when file doesn't exist."""
        result = get_unique_filename(temp_dir, "test", ".json")
        
        assert result == temp_dir / "test.json"
    
    def test_get_unique_filename_existing_file(self, temp_dir):
        """Test getting unique filename when file exists."""
        # Create existing file
        (temp_dir / "test.json").write_text("{}")
        
        result = get_unique_filename(temp_dir, "test", ".json")
        
        assert result == temp_dir / "test_1.json"
    
    def test_get_unique_filename_multiple_existing(self, temp_dir):
        """Test getting unique filename with multiple existing files."""
        # Create multiple existing files
        (temp_dir / "test.json").write_text("{}")
        (temp_dir / "test_1.json").write_text("{}")
        (temp_dir / "test_2.json").write_text("{}")
        
        result = get_unique_filename(temp_dir, "test", ".json")
        
        assert result == temp_dir / "test_3.json"
    
    def test_is_valid_path(self):
        """Test path validation."""
        assert is_valid_path("./valid/path") is True
        assert is_valid_path("C:\\Windows\\System32") is True
        assert is_valid_path("/usr/local/bin") is True
    
    def test_is_valid_path_with_path_object(self, temp_dir):
        """Test path validation with Path object."""
        assert is_valid_path(temp_dir) is True
        assert is_valid_path(temp_dir / "subdir") is True
