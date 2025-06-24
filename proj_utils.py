import matplotlib.pyplot as plt
import os
from IPython.display import HTML, display, FileLink
from datetime import datetime
import joblib

from proj_configs import MPL_STYLE_FILE, PATH_OUT_VISUALS
import base_utils_logging as log_handle


def save_and_show_link(fig_to_save, filename, base_dir=PATH_OUT_VISUALS, dpi=100):
    log_handle.logger.info("START ...")
    os.makedirs(base_dir, exist_ok=True)

    # Get absolute paths
    notebook_dir = os.getcwd()
    full_base_dir = os.path.abspath(os.path.join(notebook_dir, base_dir))
    full_filepath = os.path.join(full_base_dir, filename)

    print(f"Saving figure to {full_filepath}")

    # Save the figure with high DPI
    fig_to_save.savefig(full_filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig_to_save)  # Close the figure to free memory

    # Display a link to the saved figure
    display(FileLink(full_filepath))
    # display(HTML(full_filepath))
    log_handle.logger.info("... FINISH")

def get_current_timestamp():
    log_handle.logger.info("START ...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    log_handle.logger.info("... FINISH")
    return timestamp

def save_file(file_type_to_save, filename, base_dir_path, data):
    log_handle.logger.info("START ...")
    os.makedirs(base_dir_path, exist_ok=True)

    # Get absolute paths
    notebook_dir = os.getcwd()
    full_base_dir = os.path.abspath(os.path.join(notebook_dir, base_dir_path))
    full_filepath = os.path.join(full_base_dir, filename)

    if file_type_to_save == 'feature':
        log_handle.logger.info(f'Saving {file_type_to_save} file with filename as {filename} to path {full_filepath}')
        data.to_csv(full_filepath, index=False, header=False)
    elif file_type_to_save == 'model':
        log_handle.logger.info(f'Saving {file_type_to_save} file with filename as {filename} to path {full_filepath}')
        joblib.dump(data, full_filepath)
    elif file_type_to_save == 'hyperparams':
        log_handle.logger.info(f'Saving {file_type_to_save} file with filename as {filename} to path {full_filepath}')
        data.to_csv(full_filepath, index=False)
    elif file_type_to_save == 'metrics':
        log_handle.logger.info(f'Saving metrics into file {filename} at path {full_filepath}')
        # Check if a file exists
        if os.path.exists(full_filepath):
            # File exists, append content
            with open(full_filepath, 'a') as file:
                file.write('\n' + data)
            print(f"Content appended to existing file: {full_filepath}")
        else:
            # File doesn't exist, create and write content
            with open(full_filepath, 'w') as file:
                file.write(data)
            print(f"New file created with content: {full_filepath}")
    else:
        log_handle.logger.info(f'No matching FILE TYPE found for: {file_type_to_save}')

    log_handle.logger.info("... FINISH")