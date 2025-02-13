# Sanitize column names by:
# - Replacing spaces and special characters with underscores
# - Removing leading/trailing underscores
# - Ensuring unique column names
def sanitize_column_names(columns):
    import re
    
    sanitized = []
    seen = {}

    for col in columns:
        # Replace non-alphanumeric characters with underscores
        new_col = re.sub(r'\W+', '_', col).strip('_')

        # Ensure uniqueness
        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0

        sanitized.append(new_col)

    return(sanitized)