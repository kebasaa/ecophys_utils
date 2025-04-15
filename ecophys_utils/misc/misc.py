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
    
# Creates season labels based on 3-month seasons, not synoptic
def create_season_southern_hemisphere(timestamps):
    import warnings
    warnings.warn("Creates season labels based on 3-month seasons, not a synoptic definition", UserWarning)
    month_to_season = {
        12: "Summer", 1: "Summer", 2: "Summer",
        3: "Autumn", 4: "Autumn", 5: "Autumn",
        6: "Winter", 7: "Winter", 8: "Winter",
        9: "Spring", 10: "Spring", 11: "Spring"
    }
    return timestamps.dt.month.map(month_to_season)

def create_season_northern_hemisphere(timestamps):
    import warnings
    warnings.warn("Creates season labels based on 3-month seasons, not a synoptic definition", UserWarning)
    month_to_season = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    return timestamps.dt.month.map(month_to_season)