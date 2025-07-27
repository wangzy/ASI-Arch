import asyncio
import logging

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncAzureOpenAI

from analyse import analyse
from database import program_sample, update
from eval import evaluation
from evolve import evolve
from utils.agent_logger import end_pipeline, log_error, log_info, log_step, log_warning, start_pipeline
from utils.device_utils import setup_device_environment, get_device_info
from utils.memory_manager import get_memory_manager, start_memory_monitoring, cleanup_memory
from config import Config

client = AsyncAzureOpenAI()

set_default_openai_client(client)
set_default_openai_api("chat_completions") 

set_tracing_disabled(True)


async def run_single_experiment() -> bool:
    """Run single experiment loop - using pipeline categorized logging."""
    # Start a new pipeline process
    pipeline_id = start_pipeline("experiment")
    
    # Log device configuration
    Config.log_device_info()
    memory_manager = get_memory_manager()
    log_info(f"Memory info: {memory_manager.get_memory_info()}")
    
    try:
        # Start memory monitoring for this experiment
        with memory_manager.memory_monitor(cleanup_on_exit=True):
            # Step 1: Program sampling
            log_step("Program Sampling", "Start sampling program from database")
            context, parent = await program_sample()
            log_info(f"Program sampling completed, context length: {len(str(context))}")
            
            # Step 2: Evolution
            log_step("Program Evolution", "Start evolving new program")
            name, motivation = await evolve(context)
            if name == "Failed":
                log_error("Program evolution failed")
                end_pipeline(False, "Evolution failed")
                return False
            log_info(f"Program evolution successful, generated program: {name}")
            log_info(f"Evolution motivation: {motivation}")
            
            # Memory checkpoint after evolution
            cleanup_memory(force=False)
            
            # Step 3: Evaluation
            log_step("Program Evaluation", f"Start evaluating program {name}")
            success = await evaluation(name, motivation)
            if not success:
                log_error(f"Program {name} evaluation failed")
                end_pipeline(False, "Evaluation failed")
                return False
            log_info(f"Program {name} evaluation successful")
            
            # Memory checkpoint after evaluation
            cleanup_memory(force=False)
            
            # Step 4: Analysis
            log_step("Result Analysis", f"Start analyzing program {name} results")
            result = await analyse(name, motivation, parent=parent)
            log_info(f"Analysis completed, result: {result}")
            
            # Step 5: Update database
            log_step("Database Update", "Update results to database")
            update(result)
            log_info("Database update completed")
            
            # Successfully complete pipeline
            log_info("Experiment pipeline completed successfully")
            end_pipeline(True, f"Experiment completed successfully, program: {name}, result: {result}")
            return True
        
    except KeyboardInterrupt:
        log_warning("User interrupted experiment")
        end_pipeline(False, "User interrupted experiment")
        return False
    except Exception as e:
        log_error(f"Experiment pipeline unexpected error: {str(e)}")
        end_pipeline(False, f"Unexpected error: {str(e)}")
        return False


async def main():
    """Main function - continuous experiment execution."""
    set_tracing_disabled(True)
    
    # Initialize device environment and logging
    device, device_config = setup_device_environment()
    device_info = get_device_info()
    
    log_info("Starting continuous experiment pipeline...")
    log_info(f"Device: {device} ({device_info['device_name']})")
    log_info(f"Mixed precision: {device_config['use_mixed_precision']}")
    log_info(f"Torch compile: {device_config['use_torch_compile']}")
    
    # Start background memory monitoring
    start_memory_monitoring(device, interval=60.0)
    log_info("Background memory monitoring started")
    
    # Run plot.py first
    log_info("Running plot scripts...")
    log_info("Plot scripts completed")
    
    experiment_count = 0
    while True:
        try:
            experiment_count += 1
            log_info(f"Starting experiment {experiment_count}")
            
            # Memory cleanup before each experiment
            if experiment_count % 5 == 0:  # Every 5 experiments
                log_info("Performing periodic memory cleanup")
                cleanup_result = cleanup_memory(force=True)
                log_info(f"Memory cleanup result: {cleanup_result}")
            
            success = await run_single_experiment()
            if success:
                log_info(f"Experiment {experiment_count} completed successfully, starting next experiment...")
            else:
                log_warning(f"Experiment {experiment_count} failed, retrying in 60 seconds...")
                # Force memory cleanup on failure
                cleanup_memory(force=True)
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            log_warning("Continuous experiment interrupted by user")
            break
        except Exception as e:
            log_error(f"Main loop unexpected error: {e}")
            log_info("Retrying in 60 seconds...")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())