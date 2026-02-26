import numpy as np

from .chain import Chain

def get_row_dict(chain: Chain, params: list[str], percentiles: list[float] = [16], precision: int = None, low_error_only: bool = False) -> dict:
    """
    Get a dictionary with the mean and std of the parameters in the chain.
    
    Parameters
    ----------
    chain : Chain
        The chain from which to get the parameter statistics.
    params : list[str]
        The list of parameter names to include in the table row.
    percentiles : list[float]
        The list of quantiles to include in the table row. 
        If only one value is provided, it will be used to determine the confidence interval for the mean ± std format. 
        If two values are provided, they will be used to determine the confidence interval for the quantile format.
        Must match np.percentile format, i.e. between 0 and 100.
        Defaults to [16], which corresponds to the 16th and 84th percentiles for a 1-sigma confidence interval.
    precision : int
        The number of decimal places to use in the formatted strings. 
        If set to None, defaults to 2 significant digits of the lowest percentile.
        Defaults to None.
    low_error_only : bool
        If True, the lower error will be considered as the error for both sides. 
        Defaults to False.
    
    Returns
    -------
    dict
        A dictionary with parameter names as keys and formatted strings of their statistics as values.
        Each value is a string formatted as '$mean_{-err_low}^{+err_high}$' where mean is the mean of the parameter, 
        and err_low and err_high are the errors corresponding to the provided percentiles. 
        If both values are identical at the given precision, it will be formatted as '$mean \\pm err$'.
        
    Raises
    ------
    ValueError
        If the provided percentiles list does not contain exactly one or two values.
    """
    row = {}
    if len(percentiles) == 1:
        percentiles = [percentiles[0], 100 - percentiles[0]]
    if len(percentiles) != 2:
        raise ValueError("Percentiles must be a single value or a list of two values.")
    percentiles = np.sort(percentiles) # Ensure percentiles are in ascending order
    quantiles = np.percentile(chain.samples, percentiles, axis=0)
    
    for param in params:
        if param in chain.names:
            idx = chain.names.index(param)
            mean = chain.samples[:, idx].mean()
            err = np.abs(quantiles[:, idx] - mean)
            label = chain.labels[idx] if chain.labels else param
            
            if precision is not None:
                p = precision # Use provided precision
            else:
                # Determine it based on the lowest error to have 2 significant digits
                min_err = np.min(err)
                if min_err == 0: # Just in case of zero error
                    p = 2
                else:
                    p = max(0, int(abs(np.floor(np.log10(np.min(err)))) + 1))
            
            if round(err[0], p) == round(err[1], p) or low_error_only:
                row[label] = fr'${mean:.{p}f} \pm {err[0]:.{p}f}$'
            else:
                row[label] = fr'${mean:.{p}f}_{{-{err[0]:.{p}f}}}^{{+{err[1]:.{p}f}}}$'
    
    return row

def get_subtable_dict(*chains, **kwargs) -> dict:
    """
    Get a list of dictionaries with the mean and std of the parameters in the chains.
    
    Parameters
    ----------
    chains : list[Chain]
        The list of chains from which to get the parameter statistics.
    **kwargs
        Keyword arguments to pass to get_row_dict, such as 'params' and 'percentiles'
    
    Returns
    -------
    dict
        A dictionary with chain names as keys and row dictionaries as values.
        If the chains have a 'label' in their data, it will be used as the key, otherwise they will be named 'Chain 0', 'Chain 1', etc.
    """
    subtable = {}
    for i, chain in enumerate(chains):
        row = get_row_dict(chain, **kwargs)
        chain_name = chain.data.get('label', f'Chain {i}')
        subtable[chain_name] = row
    return subtable

def get_table(
    subtables: dict|list, 
    header_name: str = None, 
    label_dict: dict[str, str] = None,
) -> str:
    """
    Create a LaTeX-formatted table from a list of subtables.
    
    Parameters
    ----------
    subtables : dict|list
        A dictionary of subtables with their names as keys, or a list of subtables (dictionaries).
        Each subtable is a dictionary with chain names as keys and row dictionaries as values.
    header_name : str
        The name to use for the first column header (chain names).
        If None, the first column will have no header.
        Defaults to None.
    label_dict : dict[str, str]
        A dictionary mapping chain names to labels to use in the table.
        If None, the chain names will be used as-is.
        Defaults to None.
    
    Returns
    -------
    str
        A LaTeX-formatted table string. Columns will be ordered based on their first occurrence in the subtables, and missing values will be filled with "N/A".
    """
    # Define characters for simplifying the LaTeX table formatting
    tab = "\t" # Tab character for indentation
    lr = "\n" # Newline character
    tlr = " \\\\\n" # Newline character for LaTeX tables
    cs = " & " # Column separator for LaTeX tables
    hline = "\\hline" # Horizontal line for LaTeX tables
    
    if isinstance(subtables, dict):
        subtables_names = list(subtables.keys())
        subtables = list(subtables.values())
    else:
        subtables_names = None
    
    # Get the column names as the union of all parameter names in the subtables
    column_names = []
    for subtable in subtables:
        for row in subtable.values():
            column_names.extend(row.keys()) # First-seen order
    column_names = list(dict.fromkeys(column_names)) # Remove duplicates while preserving order

    # Create the LaTeX table string
    table_str = f"\\begin{{tabular}}{{{'| c ' + '| c ' * len(column_names) + '|'}}}" + lr
    
    table_str += tab + hline + lr
    
    # First column contains the chain names
    if header_name is not None: # Add header for the first column if provided
        table_str += tab + header_name + cs
    else:
        table_str += tab + cs
        
    # Add column headers
    header = cs.join([f"\\textbf{{{col}}}" for col in column_names]) + tlr
    table_str += header
    
    table_str += tab + hline + lr
    
    # Add rows for each subtable
    for i, subtable in enumerate(subtables):
        # Add subtable label if provided
        if subtables_names:
            subtable_label = subtables_names[i]
            table_str += tab + f"\\textbf{{{subtable_label}}}" + cs * len(column_names) + tlr
        # Add a row for each chain in the subtable
        for name, row in subtable.items():
            row_values = []
            row_name = label_dict[name] if label_dict and name in label_dict else name
            for col in column_names:
                row_values.append(row.get(col, "N/A"))
            table_str += tab + f"{row_name} & {cs.join(row_values)}" + tlr
        table_str += tab + hline + lr
    
    table_str += "\\end{tabular}" + lr
    return table_str