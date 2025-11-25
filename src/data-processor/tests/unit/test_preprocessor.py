import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from google.api_core.exceptions import NotFound
import sys
import os
from processor import DatasetProcessor

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# Concrete implementation of abstract class for testing
class TestDatasetProcessor(DatasetProcessor):
    """Concrete implementation of DatasetProcessor for testing purposes."""

    def filter_metadata(self, metadata, **kwargs):
        """Dummy implementation: return metadata as-is."""
        return metadata

    def format_metadata_csv(self, metadata, dataset, raw_image_path, **kwargs):
        """Dummy implementation: create properly formatted metadata."""
        formatted = pd.DataFrame(
            {
                "image_id": metadata.index.astype(str),
                "dataset": [dataset] * len(metadata),
                "filename": [f"{dataset}_{i}.jpg" for i in metadata.index],
                "orig_filename": [f"img_{i}.jpg" for i in metadata.index],
                "label": metadata.get("label", ["unknown"] * len(metadata)),
                "text_desc": metadata.get("text_desc", [None] * len(metadata)),
            }
        )
        return formatted


@pytest.fixture
def mock_storage_client():
    """Mock Google Cloud Storage client."""
    with patch("processor.storage.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def processor(mock_storage_client):
    """Create a test processor instance with mocked GCS client."""
    return TestDatasetProcessor(
        bucket_name="test-bucket",
        final_metadata_dir="final/",
        final_metadata_file="metadata_all.csv",
        final_image_path="final/imgs/",
    )


@pytest.fixture
def sample_metadata():
    """Create sample metadata DataFrame."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "label": ["eczema", "psoriasis", "eczema"],
            "text_desc": ["dry skin", "red patches", "itchy"],
            "quality_score": [0.9, 0.85, 0.95],
        }
    )


class TestDatasetProcessorInit:
    """Test DatasetProcessor initialization."""

    def test_initialization(self, processor):
        """Test that processor initializes with correct parameters."""
        # Use the fixture which has the mock already active
        assert processor.bucket_name == "test-bucket"
        assert processor.final_metadata_dir == "final/"
        assert processor.final_metadata_file == "metadata_all.csv"
        assert processor.final_image_path == "final/imgs/"

    def test_default_parameters(self, mock_storage_client):
        """Test that processor uses default parameters when not specified."""
        processor = TestDatasetProcessor()

        assert processor.bucket_name == "derma-datasets-2"
        assert processor.final_metadata_dir == "final/"
        assert processor.final_metadata_file == "metadata_all.csv"
        assert processor.final_image_path == "final/imgs/"


class TestLoadMetadata:
    """Test load_metadata function."""

    def test_load_metadata_csv(self, processor, sample_metadata, mock_storage_client):
        """Test loading metadata from CSV file."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket

        # Mock the CSV data
        csv_data = sample_metadata.to_csv(index=False).encode()
        mock_blob.download_as_bytes.return_value = csv_data

        result = processor.load_metadata("path/to/metadata.csv")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["id", "label", "text_desc", "quality_score"]
        assert len(result) == 3

    def test_load_metadata_with_kwargs(self, processor, mock_storage_client):
        """Test loading metadata with custom CSV parameters."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket

        # Create CSV with custom separator
        data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        csv_data = data.to_csv(index=False, sep=";").encode()
        mock_blob.download_as_bytes.return_value = csv_data

        result = processor.load_metadata("path/to/metadata.csv", sep=";")

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]


class TestFilterMetadata:
    """Test filter_metadata function."""

    def test_filter_metadata(self, processor, sample_metadata):
        """Test that filter_metadata returns metadata (dummy implementation)."""
        result = processor.filter_metadata(sample_metadata)

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, sample_metadata)


class TestFormatMetadataCsv:
    """Test format_metadata_csv function."""

    def test_format_metadata_csv_structure(self, processor, sample_metadata):
        """Test that formatted metadata has required columns."""
        result = processor.format_metadata_csv(sample_metadata, dataset="test_dataset", raw_image_path="raw/")

        required_columns = ["image_id", "dataset", "filename", "orig_filename", "label", "text_desc"]
        assert all(col in result.columns for col in required_columns)

    def test_format_metadata_csv_values(self, processor, sample_metadata):
        """Test that formatted metadata contains expected values."""
        result = processor.format_metadata_csv(sample_metadata, dataset="test_dataset", raw_image_path="raw/")

        assert len(result) == len(sample_metadata)
        assert all(result["dataset"] == "test_dataset")
        assert all(result["filename"].str.startswith("test_dataset_"))


class TestUpdateMetadataCsv:
    """Test _update_metadata_csv function."""

    def test_update_metadata_new_file(self, processor, sample_metadata, mock_storage_client):
        """Test creating new metadata file when it doesn't exist."""
        mock_bucket = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket

        # Mock NotFound exception when trying to load non-existent file
        with patch.object(processor, "_load_table_from_gcs", side_effect=NotFound("File not found")):
            with patch.object(processor, "_write_table_to_gcs") as mock_write:
                final_metadata = processor.format_metadata_csv(
                    sample_metadata, dataset="dataset1", raw_image_path="raw/"
                )

                processor._update_metadata_csv(final_metadata)

                mock_write.assert_called_once()
                written_df = mock_write.call_args[0][0]
                assert len(written_df) == len(final_metadata)

    def test_update_metadata_upsert(self, processor, mock_storage_client):
        """Test upserting into existing metadata file."""
        # Create old metadata
        old_metadata = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": ["dataset1", "dataset1"],
                "filename": ["dataset1_img1.jpg", "dataset1_img2.jpg"],
                "orig_filename": ["img1.jpg", "img2.jpg"],
                "label": ["eczema", "psoriasis"],
                "text_desc": ["desc1", "desc2"],
            }
        )

        # Create new metadata (overlapping dataset)
        new_metadata = pd.DataFrame(
            {
                "image_id": ["2", "3"],
                "dataset": ["dataset1", "dataset1"],
                "filename": ["dataset1_img2_new.jpg", "dataset1_img3.jpg"],
                "orig_filename": ["img2_new.jpg", "img3.jpg"],
                "label": ["psoriasis_updated", "eczema"],
                "text_desc": ["desc2_updated", "desc3"],
            }
        )

        with patch.object(processor, "_load_table_from_gcs", return_value=old_metadata):
            with patch.object(processor, "_write_table_to_gcs") as mock_write:
                processor._update_metadata_csv(new_metadata)

                written_df = mock_write.call_args[0][0]

                # Should have 3 unique entries (1, 2, 3) with 2 updated
                assert len(written_df) == 3
                # Check that id='2' was updated
                updated_row = written_df[written_df["image_id"] == "2"].iloc[0]
                assert updated_row["filename"] == "dataset1_img2_new.jpg"

    def test_update_metadata_different_datasets(self, processor, mock_storage_client):
        """Test upserting metadata from different datasets."""
        old_metadata = pd.DataFrame(
            {
                "image_id": ["1"],
                "dataset": ["dataset1"],
                "filename": ["dataset1_img1.jpg"],
                "orig_filename": ["img1.jpg"],
                "label": ["eczema"],
                "text_desc": ["desc1"],
            }
        )

        new_metadata = pd.DataFrame(
            {
                "image_id": ["1"],
                "dataset": ["dataset2"],
                "filename": ["dataset2_img1.jpg"],
                "orig_filename": ["img1.jpg"],
                "label": ["psoriasis"],
                "text_desc": ["desc_new"],
            }
        )

        with patch.object(processor, "_load_table_from_gcs", return_value=old_metadata):
            with patch.object(processor, "_write_table_to_gcs") as mock_write:
                processor._update_metadata_csv(new_metadata)

                written_df = mock_write.call_args[0][0]

                # Should have 2 entries (same image_id but different dataset)
                assert len(written_df) == 2


class TestUpdateImages:
    """Test _update_images function."""

    def test_update_images_calls_bulk_copy(self, processor, mock_storage_client):
        """Test that _update_images calls _bulk_copy_files with correct parameters."""
        image_names = ["img1.jpg", "img2.jpg", "img3.jpg"]

        with patch.object(processor, "_bulk_copy_files") as mock_copy:
            processor._update_images(image_names, "raw/images/", "dataset1")

            mock_copy.assert_called_once()
            call_args = mock_copy.call_args

            # Verify arguments
            assert call_args[0][0] == image_names
            assert call_args[0][1] == "raw/images/"
            assert call_args[0][2] == processor.final_image_path

            # Verify dest_names have correct format
            expected_dest_names = ["dataset1_img1.jpg", "dataset1_img2.jpg", "dataset1_img3.jpg"]
            assert call_args[0][3] == expected_dest_names


class TestBulkCopyFiles:
    """Test _bulk_copy_files function."""

    def test_bulk_copy_files_basic(self, processor, mock_storage_client):
        """Test basic file copying."""
        mock_bucket = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket
        mock_source_blob = MagicMock()
        mock_bucket.blob.return_value = mock_source_blob

        file_list = ["img1.jpg", "img2.jpg"]
        dest_names = ["dest1.jpg", "dest2.jpg"]

        processor._bulk_copy_files(file_list, "src/", "dst/", dest_names=dest_names, max_workers=1)

        # Verify bucket.copy_blob was called twice
        assert mock_bucket.copy_blob.call_count == 2

    def test_bulk_copy_files_length_mismatch(self, processor):
        """Test that mismatched list lengths raise ValueError."""
        with pytest.raises(ValueError, match="dest_names must be the same length"):
            processor._bulk_copy_files(
                ["img1.jpg", "img2.jpg"], "src/", "dst/", dest_names=["dest1.jpg"], max_workers=1  # Wrong length
            )

    def test_bulk_copy_files_normalizes_paths(self, processor, mock_storage_client):
        """Test that paths are normalized with trailing slashes."""
        mock_bucket = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket
        mock_source_blob = MagicMock()
        mock_bucket.blob.return_value = mock_source_blob

        file_list = ["img1.jpg"]
        dest_names = ["dest1.jpg"]

        processor._bulk_copy_files(
            file_list, "src", "dst", dest_names=dest_names, max_workers=1  # No trailing slash  # No trailing slash
        )

        # Verify source blob path has trailing slash
        blob_calls = mock_bucket.blob.call_args_list
        blob_paths = [call[0][0] for call in blob_calls if call[0]]
        assert any("src/" in path for path in blob_paths), f"Expected 'src/' in source blob path, got: {blob_paths}"

        # Verify destination path has trailing slash in copy_blob call
        copy_calls = mock_bucket.copy_blob.call_args_list
        assert len(copy_calls) > 0, "copy_blob was not called"
        dst_path = copy_calls[0][0][2]  # Third argument is destination path
        assert "dst/" in dst_path, f"Expected 'dst/' in destination path, got: {dst_path}"


class TestBulkCopyBlobs:
    """Test _bulk_copy_blobs function."""

    def test_bulk_copy_blobs_basic(self, processor, mock_storage_client):
        """Test basic blob copying."""
        mock_bucket = MagicMock()
        mock_storage_client.bucket.return_value = mock_bucket
        mock_source_blob = MagicMock()
        mock_bucket.blob.return_value = mock_source_blob

        source_blobs = ["src/img1.jpg", "src/img2.jpg"]
        dest_blobs = ["dst/img1.jpg", "dst/img2.jpg"]

        processor._bulk_copy_blobs(source_blobs, dest_blobs, max_workers=1)

        # Verify copy_blob was called twice
        assert mock_bucket.copy_blob.call_count == 2

    def test_bulk_copy_blobs_length_mismatch(self, processor):
        """Test that mismatched list lengths raise ValueError."""
        with pytest.raises(ValueError, match="source_blobs and destination_blobs must have the same length"):
            processor._bulk_copy_blobs(
                ["src/img1.jpg", "src/img2.jpg"], ["dst/img1.jpg"], max_workers=1  # Wrong length
            )


class TestListFilesInFolder:
    """Test _list_files_in_folder function."""

    def test_list_files_in_folder_basic(self, processor, mock_storage_client):
        """Test listing files in a folder."""
        mock_blobs = [MagicMock(name="img1.jpg"), MagicMock(name="img2.jpg"), MagicMock(name="subdir/")]
        mock_blobs[0].name = "folder/img1.jpg"
        mock_blobs[1].name = "folder/img2.jpg"
        mock_blobs[2].name = "folder/subdir/"

        mock_storage_client.list_blobs.return_value = mock_blobs

        result = processor._list_files_in_folder("folder/", exclude_dir=False, include_prefixes=True)

        assert len(result) == 3
        assert "folder/img1.jpg" in result

    def test_list_files_exclude_directories(self, processor, mock_storage_client):
        """Test excluding directories from file list."""
        mock_blobs = [MagicMock(name="img1.jpg"), MagicMock(name="img2.jpg"), MagicMock(name="subdir/")]
        mock_blobs[0].name = "folder/img1.jpg"
        mock_blobs[1].name = "folder/img2.jpg"
        mock_blobs[2].name = "folder/subdir/"

        mock_storage_client.list_blobs.return_value = mock_blobs

        result = processor._list_files_in_folder("folder/", exclude_dir=True, include_prefixes=True)

        assert len(result) == 2
        assert all(not item.endswith("/") for item in result)

    def test_list_files_without_prefixes(self, processor, mock_storage_client):
        """Test listing files without folder prefixes."""
        mock_blobs = [MagicMock(name="img1.jpg"), MagicMock(name="img2.jpg")]
        mock_blobs[0].name = "folder/img1.jpg"
        mock_blobs[1].name = "folder/img2.jpg"

        mock_storage_client.list_blobs.return_value = mock_blobs

        result = processor._list_files_in_folder("folder/", exclude_dir=False, include_prefixes=False)

        assert len(result) == 2
        assert "img1.jpg" in result
        assert "img2.jpg" in result

    def test_list_files_trailing_slash(self, processor, mock_storage_client):
        """Test that function adds trailing slash if needed."""
        mock_blobs = [MagicMock()]
        mock_blobs[0].name = "folder/img1.jpg"

        mock_storage_client.list_blobs.return_value = mock_blobs

        # Call without trailing slash
        processor._list_files_in_folder("folder", exclude_dir=False, include_prefixes=True)

        # Verify list_blobs was called with trailing slash
        call_args = mock_storage_client.list_blobs.call_args
        assert call_args[1]["prefix"] == "folder/"


class TestGetImgIdsNames:
    """Test _get_img_ids_names function."""

    def test_get_img_ids_names_with_prefixes(self, processor, mock_storage_client):
        """Test getting image IDs and names with prefixes included."""
        mock_files = ["raw/images/001.jpg", "raw/images/002.jpg", "raw/images/003.jpg"]

        with patch.object(processor, "_list_files_in_folder", return_value=mock_files):
            img_files, img_ids = processor._get_img_ids_names("raw/images/", include_prefixes=True)

            assert img_files == mock_files
            assert img_ids == ["001", "002", "003"]

    def test_get_img_ids_names_without_prefixes(self, processor, mock_storage_client):
        """Test getting image IDs and names without prefixes."""
        mock_files = ["001.jpg", "002.jpg", "003.jpg"]

        with patch.object(processor, "_list_files_in_folder", return_value=mock_files):
            img_files, img_ids = processor._get_img_ids_names("raw/images/", include_prefixes=False)

            assert img_files == mock_files
            assert img_ids == ["001", "002", "003"]

    def test_get_img_ids_names_trailing_slash(self, processor):
        """Test that function adds trailing slash if needed."""
        mock_files = ["001.jpg", "002.jpg"]

        with patch.object(processor, "_list_files_in_folder", return_value=mock_files) as mock_list:
            processor._get_img_ids_names("raw/images", include_prefixes=False)

            # Verify that _list_files_in_folder was called with trailing slash
            call_args = mock_list.call_args
            assert call_args[0][0] == "raw/images/"


class TestLoadTableFromGcs:
    """Test _load_table_from_gcs function."""

    def test_load_csv_from_gcs(self, processor, mock_storage_client):
        """Test loading CSV file from GCS."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket

        # Create test CSV data
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        csv_bytes = test_data.to_csv(index=False).encode()
        mock_blob.download_as_bytes.return_value = csv_bytes

        result = processor._load_table_from_gcs("path/to/file.csv")

        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, test_data)

    def test_load_parquet_from_gcs(self, processor, mock_storage_client):
        """Test loading Parquet file from GCS."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket

        # Create test data to return
        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Mock the bytes and patch pd.read_parquet to avoid needing pyarrow/fastparquet
        mock_blob.download_as_bytes.return_value = b"fake_parquet_data"

        with patch("processor.pd.read_parquet", return_value=test_data) as mock_read:
            result = processor._load_table_from_gcs("path/to/file.parquet")

            # Verify pd.read_parquet was called with BytesIO object
            mock_read.assert_called_once()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3

    def test_load_unsupported_format(self, processor, mock_storage_client):
        """Test that unsupported file formats raise ValueError."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket
        mock_blob.download_as_bytes.return_value = b"data"

        with pytest.raises(ValueError, match="Unsupported file extension"):
            processor._load_table_from_gcs("path/to/file.unknown")


class TestWriteTableToGcs:
    """Test _write_table_to_gcs function."""

    def test_write_csv_to_gcs(self, processor, mock_storage_client):
        """Test writing CSV file to GCS."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket

        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        processor._write_table_to_gcs(test_data, "path/to/output.csv")

        # Verify blob.upload_from_file was called
        mock_blob.upload_from_file.assert_called_once()
        call_args = mock_blob.upload_from_file.call_args

        # Verify content type
        assert call_args[1]["content_type"] == "text/csv"

    def test_write_csv_with_kwargs(self, processor, mock_storage_client):
        """Test writing CSV with custom parameters."""
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket

        test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        processor._write_table_to_gcs(test_data, "path/to/output.csv", sep=";")

        # Verify upload was called
        mock_blob.upload_from_file.assert_called_once()


class TestUpdateData:
    """Test update_data function."""

    def test_update_data_calls_all_helpers(self, processor, sample_metadata, mock_storage_client):
        """Test that update_data calls all helper functions."""
        final_metadata = processor.format_metadata_csv(sample_metadata, dataset="test_dataset", raw_image_path="raw/")

        with patch.object(processor, "_update_metadata_csv") as mock_update_meta:
            with patch.object(processor, "_update_images") as mock_update_imgs:
                with patch.object(processor, "_write_table_to_gcs") as mock_write:
                    processor.update_data(final_metadata, sample_metadata, "filtered_metadata.csv", "raw/test_dataset/")

                    # Verify all helpers were called
                    mock_update_meta.assert_called_once()
                    mock_update_imgs.assert_called_once()
                    mock_write.assert_called_once()


class TestAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that DatasetProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DatasetProcessor()

    def test_concrete_implementation_works(self, mock_storage_client):
        """Test that concrete implementation can be instantiated."""
        processor = TestDatasetProcessor()
        assert isinstance(processor, DatasetProcessor)
