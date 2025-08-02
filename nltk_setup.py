import nltk
import ssl

def download_nltk_data():
    """
    Checks for and downloads required NLTK data packages.
    """
    try:
        # This is a workaround for an SSL certificate issue on some systems
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # List of packages to check and download
    required_packages = ['punkt', 'punkt_tab']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"Downloading NLTK '{package}' package...")
            nltk.download(package, quiet=True)
            print(f"'{package}' downloaded.")