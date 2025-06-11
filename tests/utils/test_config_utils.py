import os
import pytest
import yaml
import tempfile

from ml_training_base import load_config

# --- Fixtures ---

@pytest.fixture
def valid_config_file() -> str:
    """
    Creates a temporary, valid YAML file and yields its path.
    """
    config_data = {
        "data": {"path": "/data/sets", "batch_size": 32},
        "model": {"name": "ResNet50", "learning_rate": 0.001}
    }
    # Use tempfile to handle file creation and cleanup
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_data, tmp)
        tmp_path = tmp.name

    yield tmp_path

    os.remove(tmp_path)


@pytest.fixture
def invalid_config_file() -> str:
    """
    Creates a temporary file with invalid YAML syntax.
    """
    invalid_yaml_content = "key: [value1, value2"
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as tmp:
        tmp.write(invalid_yaml_content)
        tmp_path = tmp.name

    yield tmp_path

    os.remove(tmp_path)

# --- Test Functions ---

def test_load_config_success(valid_config_file: str):
    """
    Tests that a valid YAML configuration file is loaded correctly into a dictionary.
    """
    # Act
    config = load_config(valid_config_file)

    # Assert
    assert isinstance(config, dict)
    assert "data" in config
    assert "model" in config
    assert config["data"]["batch_size"] == 32
    assert config["model"]["name"] == "ResNet50"


def test_load_config_file_not_found():
    """
    Tests that a FileNotFoundError is raised if the config file does not exist.
    """
    # Arrange
    non_existent_path = "path/to/non/existent/config.yaml"

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config(non_existent_path)


def test_load_config_invalid_yaml(invalid_config_file: str):
    """
    Tests that a YAMLError is raised if the file contains invalid YAML syntax.
    """
    # Act & Assert
    with pytest.raises(yaml.YAMLError, match="Error parsing YAML file"):
        load_config(invalid_config_file)
