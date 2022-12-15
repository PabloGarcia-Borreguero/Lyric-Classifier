import re




# function to clean data
def clean_data(df, col, clean_col):
    """_summary_

    Args:
        df (_type_): _description_
        col (_type_): _description_
        clean_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    # change to lower and remove spaces on either side
    df[clean_col] = df[col].apply(lambda x: x.lower().strip())

    # remove extra spaces in between
    df[clean_col] = df[clean_col].apply(lambda x: re.sub(' +', ' ', x))

    # remove unwanted delimeters
    df[clean_col] = df[clean_col].apply(lambda x: x.replace("<br />", ""))

    # remove punctuation
    df[clean_col] = df[clean_col].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

    return df
