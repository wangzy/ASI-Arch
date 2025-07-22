from datetime import datetime
from typing import List, Tuple

from agents import exceptions
from config import Config
from database.mongo_database import create_client
from utils.agent_logger import log_agent_run
from .model import code_checker, creator, motivation_checker, optimizer
from .prompt import (
    CodeChecker_input,
    Creator_input,
    Creator_input_dedup,
    Motivation_checker_input,
    Optimizer_input,
    Optimizer_input_dedup
)


async def evolve(context: str, mode: str) -> Tuple[str, str]:
    """
    Evolve code based on context and mode.
    
    Args:
        context: Context for evolution
        mode: Evolution mode ('create' or 'optimize')
        
    Returns:
        Tuple of (name, motivation) or error information
    """
    if mode == 'create':
        return await create(context)
    elif mode == 'optimize':
        return await optimize(context)
    else:
        return "Failed", "mode error"


async def create(context: str) -> Tuple[str, str]:
    """
    Create new code variant based on context.
    
    Args:
        context: Context for creation
        
    Returns:
        Tuple of (timestamped_name, motivation) or error information
    """
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        with open(Config.SOURCE_FILE, 'r', encoding='utf-8') as f:
            original_source = f.read()
            
        name, motivation = await creating(context)
        
        # Generate timestamp prefix (format: YYYYMMDD-HH:MM:SS)
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        
        # Add timestamp prefix to name
        timestamped_name = f"{timestamp}-{name}"
        
        if await check_code_correctness(motivation):
            return timestamped_name, motivation

        with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
            f.write(original_source)
        print("Try new motivations")
    return "Failed", "create error"


async def optimize(context: str) -> Tuple[str, str]:
    """
    Optimize existing code based on context.
    
    Args:
        context: Context for optimization
        
    Returns:
        Tuple of (timestamped_name, motivation) or error information
    """
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        with open(Config.SOURCE_FILE, 'r', encoding='utf-8') as f:
            original_source = f.read()
            
        name, motivation = await optimizing(context)
        
        # Generate timestamp prefix (format: YYYYMMDD-HH:MM:SS)
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        
        # Add timestamp prefix to name
        timestamped_name = f"{timestamp}-{name}"
        
        if await check_code_correctness(motivation):
            return timestamped_name, motivation

        with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
            f.write(original_source)
        print("Try new motivations")
    return "Failed", "optimize error"


async def creating(context: str) -> Tuple[str, str]:
    """
    Create new code with deduplication retry mechanism.
    
    Args:
        context: Context for creation
        
    Returns:
        Tuple of (name, motivation)
        
    Raises:
        Exception: When maximum retry attempts reached or other errors occur
    """
    # Save original file content
    with open(Config.SOURCE_FILE, 'r', encoding='utf-8') as f:
        original_source = f.read()
        
    repeated_result = None
    motivation = None
    
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            # Restore original file
            with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
                f.write(original_source)
            
            # Use different prompts based on repetition status
            plan = None
            if attempt == 0:
                input_data = Creator_input(context)
                plan = await log_agent_run("creator", creator, input_data)
            else:
                repeated_context = get_repeated_context(repeated_result.repeated_index)
                input_data = Creator_input_dedup(context, repeated_context)
                plan = await log_agent_run("creator", creator, input_data)
                
            name, motivation = plan.final_output.name, plan.final_output.motivation
            
            repeated_result = await check_repeated_motivation(motivation)
            if repeated_result.is_repeated:
                print(
                    f"Attempt {attempt + 1}: Motivation repeated, "
                    f"index: {repeated_result.repeated_index}"
                )
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    raise Exception(
                        "Maximum retry attempts reached, unable to generate non-repeated motivation"
                    )
                continue
            else:
                print(f"Attempt {attempt + 1}: Motivation not repeated, continuing execution")
                print(motivation)
                return name, motivation
                
        except exceptions.MaxTurnsExceeded as e:
            print(f"Attempt {attempt + 1}: Exceeded maximum dialogue turns")
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}")
            raise e


async def optimizing(context: str) -> Tuple[str, str]:
    """
    Optimize code with deduplication retry mechanism.
    
    Args:
        context: Context for optimization
        
    Returns:
        Tuple of (name, motivation)
        
    Raises:
        Exception: When maximum retry attempts reached or other errors occur
    """
    # Save original file content
    with open(Config.SOURCE_FILE, 'r', encoding='utf-8') as f:
        original_source = f.read()
        
    repeated_result = None
    motivation = None
    
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            # Restore original file
            with open(Config.SOURCE_FILE, 'w', encoding='utf-8') as f:
                f.write(original_source)
            
            # Use different prompts based on repetition status
            plan = None
            if attempt == 0:
                input_data = Optimizer_input(context)
                plan = await log_agent_run("optimizer", optimizer, input_data)
            else:
                repeated_context = get_repeated_context(repeated_result.repeated_index)
                input_data = Optimizer_input_dedup(context, repeated_context)
                plan = await log_agent_run("optimizer", optimizer, input_data)
                
            name, motivation = plan.final_output.name, plan.final_output.motivation
            
            repeated_result = await check_repeated_motivation(motivation)
            if repeated_result.is_repeated:
                print(
                    f"Attempt {attempt + 1}: Motivation repeated, "
                    f"index: {repeated_result.repeated_index}"
                )
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    raise Exception(
                        "Maximum retry attempts reached, unable to generate non-repeated motivation"
                    )
                continue
            else:
                print(f"Attempt {attempt + 1}: Motivation not repeated, continuing execution")
                print(motivation)
                return name, motivation
                
        except exceptions.MaxTurnsExceeded as e:
            print(f"Attempt {attempt + 1}: Exceeded maximum dialogue turns")
        except Exception as e:
            print(f"Attempt {attempt + 1} error: {e}")
            raise e


async def check_code_correctness(motivation: str) -> bool:
    """
    Check code correctness using code checker.
    
    Args:
        motivation: Motivation for the code check
        
    Returns:
        True if code is correct, False otherwise
    """
    for attempt in range(Config.MAX_RETRY_ATTEMPTS):
        try:
            code_checker_result = await log_agent_run(
                "code_checker",
                code_checker,
                CodeChecker_input(motivation=motivation),
                max_turns=100
            )
            
            if code_checker_result.final_output.success:
                print("Code checker passed - code looks correct")
                return True
            else:
                error_msg = code_checker_result.final_output.error
                print(f"Code checker found issues: {error_msg}")
                if attempt == Config.MAX_RETRY_ATTEMPTS - 1:
                    print("Reaching checking limits")
                    return False
                continue
                
        except exceptions.MaxTurnsExceeded as e:
            print("Code checker exceeded maximum turns")
            return False
        except Exception as e:
            print(f"Code checker error: {e}")
            return False


async def check_repeated_motivation(motivation: str):
    """
    Check if motivation is repeated using similarity search.
    
    Args:
        motivation: Motivation text to check
        
    Returns:
        Result indicating if motivation is repeated
    """
    client = create_client()
    similar_elements = client.search_similar_motivations(motivation)
    context = similar_motivation_context(similar_elements)
    input_data = Motivation_checker_input(context, motivation)
    repeated_result = await log_agent_run("motivation_checker", motivation_checker, input_data)
    return repeated_result.final_output


def similar_motivation_context(similar_elements: list) -> str:
    """
    Generate structured context from similar motivation elements.
    
    Args:
        similar_elements: List of similar motivation elements
        
    Returns:
        Formatted context string
    """
    if not similar_elements:
        return "No previous motivations found for comparison."
    
    context = "### PREVIOUS RESEARCH MOTIVATIONS\n\n"
    
    for i, element in enumerate(similar_elements, 1):
        context += f"**Reference #{i} (Index: {element.index})**\n"
        context += f"```\n{element.motivation}\n```\n\n"
    
    context += f"**Total Previous Motivations**: {len(similar_elements)}\n"
    context += "**Analysis Scope**: Compare target motivation against each reference above\n"
    
    return context


def get_repeated_context(repeated_index: List[int]) -> str:
    """
    Generate structured context from repeated motivation experiments.
    
    Args:
        repeated_index: List of indices for repeated experiments
        
    Returns:
        Formatted context string for repeated experiments
    """
    client = create_client()
    repeated_elements = [client.get_elements_by_index(index) for index in repeated_index]
    
    if not repeated_elements:
        return "No repeated experimental context available."
    
    structured_context = "### REPEATED EXPERIMENTAL PATTERNS ANALYSIS\n\n"
    
    for i, element in enumerate(repeated_elements, 1):
        structured_context += f"**Experiment #{i} - Index {element.index}**\n"
        structured_context += f"```\n{element.motivation}\n```\n\n"
    
    structured_context += "**Pattern Analysis Summary:**\n"
    structured_context += f"- **Total Repeated Experiments**: {len(repeated_elements)}\n"
    structured_context += (
        "- **Innovation Challenge**: Break free from these established pattern spaces\n"
    )
    structured_context += (
        "- **Differentiation Requirement**: Implement orthogonal approaches that explore "
        "fundamentally different design principles\n\n"
    )
    
    structured_context += (
        "**Key Insight**: The above experiments represent exhausted design spaces. "
        "Your task is to identify and implement approaches that operate on completely "
        "different mathematical, biological, or physical principles to achieve "
        "breakthrough innovation.\n"
    )
    
    return structured_context