from setuptools import setup, find_packages

def get_requirements(file_path):
    """
    This function returns a list of requirements from the given file path.
    """
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements
   

setup(
    name="my_package",
    version="0.0.1",
    author="Hritik Shukla",
    author_email="shritik83385@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
