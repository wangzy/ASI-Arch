from utils.device_utils import setup_device_environment, get_device_info, get_device_specific_config
import logging

# Setup device environment
DEVICE, DEVICE_CONFIG = setup_device_environment()
DEVICE_INFO = get_device_info()

class Config:
    """Configuration settings for the experiment."""
    # Target file
    SOURCE_FILE: str = "evolve file"
    
    # Training script
    BASH_SCRIPT: str = "your training script"
    
    # Experiment results
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"
    
    # Debug file
    DEBUG_FILE: str = "./files/debug/training_error.txt"
    
    # Code pool directory
    CODE_POOL: str = "./pool"
    
    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 3
    
    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 10
    
    # RAG service URL
    RAG: str = "your rag url"
    
    # Database URL
    DATABASE: str = "your database url"
    
    # Device Configuration (Auto-detected)
    DEVICE = DEVICE
    DEVICE_TYPE: str = DEVICE_INFO["device_type"]
    DEVICE_NAME: str = DEVICE_INFO["device_name"]
    
    # Device-specific training settings
    USE_MIXED_PRECISION: bool = DEVICE_CONFIG["use_mixed_precision"]
    USE_TORCH_COMPILE: bool = DEVICE_CONFIG["use_torch_compile"]
    BATCH_SIZE_MULTIPLIER: float = DEVICE_CONFIG["batch_size_multiplier"]
    GRADIENT_ACCUMULATION_STEPS: int = DEVICE_CONFIG["gradient_accumulation_steps"]
    
    # Memory and performance settings
    PIN_MEMORY: bool = DEVICE_CONFIG["pin_memory"]
    NUM_WORKERS: int = DEVICE_CONFIG["num_workers"]
    NON_BLOCKING: bool = DEVICE_CONFIG["non_blocking"]
    
    @classmethod
    def log_device_info(cls):
        """Log device configuration information."""
        logger = logging.getLogger(__name__)
        logger.info(f"ASI-Arch Device Configuration:")
        logger.info(f"  Device: {cls.DEVICE}")
        logger.info(f"  Type: {cls.DEVICE_TYPE}")
        logger.info(f"  Name: {cls.DEVICE_NAME}")
        logger.info(f"  Mixed Precision: {cls.USE_MIXED_PRECISION}")
        logger.info(f"  Torch Compile: {cls.USE_TORCH_COMPILE}")
        logger.info(f"  Batch Size Multiplier: {cls.BATCH_SIZE_MULTIPLIER}")
        logger.info(f"  Gradient Accumulation: {cls.GRADIENT_ACCUMULATION_STEPS}")