# AI-Jobs
import nltk
nltk.download('wordnet')
nltk.download('punkt')  # Needed for tokenization
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud  # Import WordCloud for visualization

# Load your cleaned DataFrame with semicolon as delimiter
file_path = r"C:\Users\Dell\Desktop\parsed_job_data.csv"

# Load the DataFrame with the appropriate separator and low_memory option
df = pd.read_csv(file_path, sep=';', low_memory=False, on_bad_lines='skip')

# Check the first few rows to identify any issues
print(df.head())

# Display the total rows and columns loaded
print("Total rows and columns loaded:", df.shape)

# Clean up the column names by stripping whitespace
df.columns = df.columns.str.replace(',', '').str.strip()

# Re-check the number of columns after cleaning
print("Number of columns after cleaning:", df.shape[1])

# Ensure that you have the expected number of columns before reassigning names
expected_columns = 8  # Change this based on your dataset structure

if df.shape[1] == expected_columns:
    df.columns = ["Job_title", "Company_name", "Location", "Salary_fork", 
                   "Rating", "Company_Overview", "Job_description", "Avg_base_salary"]
else:
    print(f"Warning: Expected {expected_columns} columns but got {df.shape[1]}.")

# Remove unnecessary rows and reset index
df.drop(index=[1, 2, 3, 4], inplace=True)  # Adjust based on your data
df.reset_index(drop=True, inplace=True)

# Remove NaN values
df.dropna(inplace=True)

# Convert 'Salary_fork' to numeric
df['Salary_fork'] = df['Salary_fork'].str.replace(',', '').str.extract(r'(\d+)')[0]
df['Salary_fork'] = pd.to_numeric(df['Salary_fork'], errors='coerce')

# Clean the 'Avg_base_salary' column
df['Avg_base_salary'] = df['Avg_base_salary'].str.replace(',', '').str.strip()  # Remove commas and strip whitespace
df['Avg_base_salary'] = pd.to_numeric(df['Avg_base_salary'], errors='coerce')  # Convert to numeric

# Check data types and summary info
print(df.info())

# Remove duplicate rows
df.drop_duplicates(subset=['Job_title', 'Company_name'], inplace=True)

# Display total rows after removing duplicates
print("Total rows after removing duplicates:", df.shape[0])

# Data Cleaning Section
# Initialize stemmer, lemmatizer, and stop words
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 1. Minimum number of employees for each listed company using information from the Company_Overview column
def extract_min_employees(overview):
    match = re.search(r'(\d+)\s*to\s*(\d+)\s*Employees', overview)
    if match:
        return int(match.group(1))  # Return minimum number
    return None

df['Min_Employees'] = df['Company_Overview'].apply(extract_min_employees)

# 2. Identify the sector or industry focus for each tech company based on the details in the Company_Overview column
def extract_industry(overview):
    industries = {
        'Software': r'\bsoftware\b',
        'Finance': r'\bfinance\b',
        'Healthcare': r'\bhealthcare\b',
        'Retail': r'\bretail\b',
        'Telecom': r'\btelecom\b',
        'Education': r'\beducation\b',
        'Automotive': r'\bautomotive\b'
    }
    
    for industry, pattern in industries.items():
        if re.search(pattern, overview, re.IGNORECASE):
            return industry
    return 'Other'

df['Industry'] = df['Company_Overview'].apply(extract_industry)

# 3. Extract the year each company was founded from the Company_Overview column
def extract_foundation_year(overview):
    match = re.search(r'\b(19|20)\d{2}\b', overview)
    if match:
        return int(match.group(0))  # Return the year found
    return None

df['Foundation_Year'] = df['Company_Overview'].apply(extract_foundation_year)

# Visualize the distribution of Average Base Salary
df['Avg_base_salary'].hist(bins=20)
plt.title('Distribution of Average Base Salary')
plt.xlabel('Average Base Salary')
plt.ylabel('Frequency')
plt.show()

# Save the cleaned DataFrame
df.to_csv('cleaned_job_data.csv', index=False)

# Job Description Cleanup

# Function to remove words shorter than 3 characters and stop words
def remove_short_words_and_stop(text):
    words = text.split()
    cleaned_text = " ".join([word for word in words if len(word) >= 3 and word.lower() not in stop_words])
    return cleaned_text

# Ensure minimum length of job descriptions
max_length = df['Job_description'].str.len().max()
min_length = max_length * 0.05

# Apply minimum length filter
df['Job_description'] = df['Job_description'].apply(lambda x: x if len(x) >= min_length else "")

# Tokenize, stem, and lemmatize job descriptions
def preprocess_text(text):
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Stem
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    # Remove short words and stop words
    cleaned_tokens = [token for token in lemmatized_tokens if len(token) >= 3 and token.lower() not in stop_words]
    return " ".join(cleaned_tokens)

df['Job_description'] = df['Job_description'].apply(preprocess_text)

# Calculate average salary for AI jobs
def is_ai_job(job_title):
    # Define keywords or regular expressions to identify AI jobs
    keywords = ["machine learning", "artificial intelligence", "deep learning", "computer vision", "natural language processing"]
    for keyword in keywords:
        if keyword.lower() in job_title.lower():
            return True
    return False

# Filter AI jobs based on Job_title
ai_jobs_df = df[df['Job_title'].apply(is_ai_job)]

# Calculate average salary for AI jobs (assuming Salary_fork is already numeric)
avg_salary_ai = ai_jobs_df['Salary_fork'].mean()
print(f"Average salary for AI jobs: ${round(avg_salary_ai, 2)}")

# Determine salary range (min and max) from Salary_fork
def extract_salary_range(salary_fork):
    if pd.isna(salary_fork):
        return None, None
    try:
        min_salary, max_salary = salary_fork.split(" - ")
        min_salary = int(min_salary.strip().replace(",", ""))
        max_salary = int(max_salary.strip().replace(",", ""))
        return min_salary, max_salary
    except Exception as e:
        print(f"Error extracting salary range: {e}")
        return None, None

# Apply the function to Salary_fork column
df['Min_Salary'], df['Max_Salary'] = zip(*df['Salary_fork'].apply(extract_salary_range))

# Get min and max salary for all jobs
min_salary_all = df['Min_Salary'].min()
max_salary_all = df['Max_Salary'].max()
print(f"Salary Range (all jobs): ${min_salary_all} - ${max_salary_all}")

# Get min and max salary for AI jobs (optional)
min_salary_ai = ai_jobs_df['Min_Salary'].min()
max_salary_ai = ai_jobs_df['Max_Salary'].max()
print(f"Salary Range (AI jobs): ${min_salary_ai} - ${max_salary_ai}")

# Convert the 'Rating' column to integers, rounding to the nearest integer
df['Rating'] = df['Rating'].apply(lambda x: round(x))

# Create a final DataFrame with the desired columns
final_columns = {
    'Job_title': 'job_title',  # Renaming for final output
    'Company_name': 'company_name',
    'Location': 'location',
    'Job_description': 'job_description',  # Already cleaned and processed
    'Min_Employees': 'employees',  # Minimum number of employees
    'Foundation_Year': 'founded',  # Year the company was founded
    'Industry': 'sector',  # Sector of the company
    'Avg_base_salary': 'avg_salary',  # Average salary
    'Min_Salary': 'salary_range_min',  # Minimum salary for the job
    'Max_Salary': 'salary_range_max',  # Maximum salary for the job
    'Rating': 'rating'  # Rounded rating
}  # Closing the dictionary
