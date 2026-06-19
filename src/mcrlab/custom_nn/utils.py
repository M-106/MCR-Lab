# -----------
# > Imports <
# -----------



# -----------
# > Logging <
# -----------
def extract_values(dict_, top_name=None):
    flattened_dict = {}
    
    for name, value in dict_.items():
        # Create the nested key name -> e.g., "config_for_logging.learning_rate"
        current_name = f"{top_name}.{name}" if top_name else name
        
        if isinstance(value, dict):
            # Recursively flatten nested dictionaries
            flattened_dict.update(extract_values(value, top_name=current_name))
            
        elif isinstance(value, (list, tuple)):
            # Convert lists and tuples to strings for TensorBoard
            flattened_dict[current_name] = str(value)
            
        elif value is None:
            # Convert NoneType to a string
            flattened_dict[current_name] = "None"
            
        else:
            # Add primitive types (int, float, bool, str) directly
            flattened_dict[current_name] = value
            
    return flattened_dict

def log_config(writer, **kwargs):
    flat_hparams = extract_values(kwargs)
    writer.add_hparams(hparam_dict=flat_hparams, metric_dict={})







