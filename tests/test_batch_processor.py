
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.tools.batch_processor import batch_process_images
from src.tools.base import BatchProcessingResult

class TestBatchProcessImagesStatus:
    
    @pytest.fixture
    def mock_find_images(self):
        with patch("src.tools.batch_processor.find_images_in_directory") as mock:
            yield mock

    @pytest.fixture
    def mock_load_existing(self):
        with patch("src.tools.batch_processor.load_existing_questions") as mock:
            yield mock

    def test_status_completed(self, tmp_path, mock_find_images, mock_load_existing):
        # Setup: 2 images found, both processed
        mock_find_images.return_value = [str(tmp_path / "img1.jpg"), str(tmp_path / "img2.jpg")]
        
        # Mock existing questions with processed_images
        mock_load_existing.return_value = (
            {
                "multiple_choice": [], 
                "true_false": [], 
                "processed_images": [str(tmp_path / "img1.jpg"), str(tmp_path / "img2.jpg")]
            }, 
            None
        )
        
        # Create dummy file so it exists
        (tmp_path / "questions.json").touch()
        
        with patch("src.tools.batch_processor.BatchProcessingResult") as mock_result_cls:
            # We need to make sure the mock returns a valid object that has attributes used in the function
            mock_instance = MagicMock()
            mock_instance.total_images = 2
            mock_instance.processed_images = ["img1.jpg", "img2.jpg"]
            mock_instance.unprocessed_images = []
            mock_result_cls.return_value = mock_instance
            
            batch_process_images.invoke({
                "directory_path": str(tmp_path),
                "recursive": False
            })
            
            # Verify BatchProcessingResult was initialized with status="completed"
            call_args = mock_result_cls.call_args[1]
            assert call_args["status"] == "completed"

    def test_status_pending(self, tmp_path, mock_find_images, mock_load_existing):
        # Setup: 2 images found, none processed
        mock_find_images.return_value = [str(tmp_path / "img1.jpg"), str(tmp_path / "img2.jpg")]
        
        # Mock existing questions (none processed)
        mock_load_existing.return_value = (
            {"multiple_choice": [], "true_false": [], "processed_images": []}, 
            None
        )
        
        (tmp_path / "questions.json").touch()
        
        with patch("src.tools.batch_processor.BatchProcessingResult") as mock_result_cls:
            mock_instance = MagicMock()
            mock_instance.total_images = 2
            mock_instance.processed_images = []
            mock_instance.unprocessed_images = ["img1.jpg", "img2.jpg"]
            mock_result_cls.return_value = mock_instance
            
            batch_process_images.invoke({
                "directory_path": str(tmp_path),
                "recursive": False
            })
            
            call_args = mock_result_cls.call_args[1]
            assert call_args["status"] == "pending"

    def test_status_in_progress(self, tmp_path, mock_find_images, mock_load_existing):
        # Setup: 2 images found, 1 processed
        mock_find_images.return_value = [str(tmp_path / "img1.jpg"), str(tmp_path / "img2.jpg")]
        
        # Mock existing questions (1 processed)
        mock_load_existing.return_value = (
            {
                "multiple_choice": [], 
                "true_false": [], 
                "processed_images": [str(tmp_path / "img1.jpg")]
            }, 
            None
        )
        
        (tmp_path / "questions.json").touch()
        
        with patch("src.tools.batch_processor.BatchProcessingResult") as mock_result_cls:
            mock_instance = MagicMock()
            mock_instance.total_images = 2
            mock_instance.processed_images = ["img1.jpg"]
            mock_instance.unprocessed_images = ["img2.jpg"]
            mock_result_cls.return_value = mock_instance
            
            batch_process_images.invoke({
                "directory_path": str(tmp_path),
                "recursive": False
            })
            
            call_args = mock_result_cls.call_args[1]
            assert call_args["status"] == "in_progress"

    def test_output_format_batching(self, tmp_path, mock_find_images, mock_load_existing):
        # Setup: 5 images found, none processed
        images = [str(tmp_path / f"img{i}.jpg") for i in range(1, 6)]
        mock_find_images.return_value = images
        
        # Mock existing questions (none processed)
        mock_load_existing.return_value = (
            {"multiple_choice": [], "true_false": [], "processed_images": []}, 
            None
        )
        
        (tmp_path / "questions.json").touch()
        
        # We do NOT mock BatchProcessingResult here because we want to test the output string generation
        # which relies on the real object's data.
        
        output = batch_process_images.invoke({
            "directory_path": str(tmp_path),
            "recursive": False,
            "batch_size": 2
        })
        
        # Verify output contains "Next Batch to Process"
        assert "Next Batch to Process (Batch Size: 2):" in output
        
        # Verify only 2 images are listed
        # The tool uses the paths returned by find_images_in_directory
        assert f"- {str(tmp_path / 'img1.jpg')}" in output
        assert f"- {str(tmp_path / 'img2.jpg')}" in output
        assert f"- {str(tmp_path / 'img3.jpg')}" not in output
        
        # Verify "more pending images not shown" message
        assert "... and 3 more pending images not shown." in output
        
        # Verify recommended actions
        assert "Call `analyze_image` with the images listed in 'Next Batch to Process' above." in output
