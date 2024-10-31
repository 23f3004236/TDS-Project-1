# Data Scraping #

### Analysis Steps ###

- **Authentication and Rate Limit Handling:** The code starts with setting up an authentication header using a GitHub personal access token. A function, `check_rate_limit`, is included to monitor and handle GitHub's API rate limits by pausing execution until the rate limit resets.

- **Data Cleaning:** The `clean_company_name` function standardizes company names by removing leading "`@`" symbols and converting them to uppercase.

- **Fetching Sydney Users:** The function `get_sydney_users` retrieves GitHub users based in Sydney with over 100 followers. For each user, it sends a request to GitHub’s search API, follows pagination to gather as many users as possible, and pauses briefly to avoid rate limits. User details like login, name, company, location, and bio are stored in a list.

- **Fetching Repositories for Each User:** The `get_user_repositories` function collects repositories for each user retrieved in `get_sydney_users`. This function retrieves details such as repository name, creation date, star count, language, and license type. The loop stops after gathering up to `500` repositories.

- **Writing to CSV Files:** The functions `write_users_csv` and `write_repositories_csv` save the user and repository data into two CSV files, `users.csv` and `repositories.csv`, with defined column names for each data point.

- **Main Execution:** In the `main` function, the code orchestrates the entire process:
   - Fetches Sydney-based users.
   - Writes the user data to `users.csv`.
   - Iterates over each user to fetch their repositories and saves the data into `repositories.csv`.


### Complete Code ###

```
import requests
import csv
import time
from datetime import datetime

access_token = '<my_GitHub_personal_access_token>'
headers = {'Authorization': f'token {access_token}'}

def check_rate_limit(response):
    if int(response.headers.get('X-RateLimit-Remaining', 0)) == 0:
        wait_time = int(response.headers.get('X-RateLimit-Reset', 0)) - int(time.time()) + 5
        time.sleep(wait_time)
        return True
    return False

def clean_company_name(company):
    return company.strip('@').upper() if company else ''

def fetch_data(url):
    response = requests.get(url, headers=headers)
    if check_rate_limit(response):
        response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else None

def get_sydney_users():
    users, page = [], 1
    while True:
        data = fetch_data(f"https://api.github.com/search/users?q=location:Sydney+followers:>100&per_page=100&page={page}")
        if not data or 'items' not in data:
            break
        for user in data['items']:
            user_info = fetch_data(user.get('url'))
            if user_info:
                users.append({
                    'login': user_info.get('login', ''),
                    'name': user_info.get('name', ''),
                    'company': clean_company_name(user_info.get('company', '')),
                    'location': user_info.get('location', ''),
                    'email': user_info.get('email', ''),
                    'hireable': 'true' if user_info.get('hireable') else 'false',
                    'bio': user_info.get('bio', ''),
                    'public_repos': user_info.get('public_repos', 0),
                    'followers': user_info.get('followers', 0),
                    'following': user_info.get('following', 0),
                    'created_at': user_info.get('created_at', '')
                })
        page += 1
        time.sleep(1)
    return users

def get_user_repositories(user_login):
    repos, page = [], 1
    while len(repos) < 500:
        repo_data = fetch_data(f"https://api.github.com/users/{user_login}/repos?per_page=100&page={page}&sort=pushed")
        if not repo_data:
            break
        for repo in repo_data:
            license_info = repo.get('license', {}).get('name', '') if repo.get('license', {}).get('key') != 'other' else ''
            repos.append({
                'login': user_login,
                'full_name': repo.get('full_name', ''),
                'created_at': repo.get('created_at', ''),
                'stargazers_count': repo.get('stargazers_count', 0),
                'watchers_count': repo.get('watchers_count', 0),
                'language': repo.get('language', ''),
                'has_projects': 'true' if repo.get('has_projects') else 'false',
                'has_wiki': 'true' if repo.get('has_wiki') else 'false',
                'license_name': license_info
            })
        page += 1
        time.sleep(1)
    return repos

def write_csv(filename, data, fieldnames):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def main():
    users = get_sydney_users()
    write_csv('users.csv', users, ['login', 'name', 'company', 'location', 'email', 'hireable', 'bio', 'public_repos', 'followers', 'following', 'created_at'])
    all_repositories = [repo for user in users for repo in get_user_repositories(user['login'])]
    write_csv('repositories.csv', all_repositories, ['login', 'full_name', 'created_at', 'stargazers_count', 'watchers_count', 'language', 'has_projects', 'has_wiki', 'license_name'])

if __name__ == "__main__":
    main()

```


# Data Analysis #

### Setup and Analysis Steps ###

- **Important import libraries:**
  - `train_test_split`: Splits data into training and testing sets.
  - `LogisticRegression`: Builds a logistic regression model to predict outcomes.
  - `classification_report`: Generates a performance summary of the model.
  - `numpy`: Used for numerical operations (although not directly here).

- **Setting Engagement** Threshold: Define an engagement threshold of `100` stargazers, meaning any repository with over 100 stargazers will be considered `"high engagement"`.

- **Creating a Binary Target Variable:** Add a new column `high_engagement` to `repositories_df`. Set it to 1 if `stargazers_count` is above 100, otherwise 0. This will be our target for predicting high engagement.

- **Selecting and Preparing Features:** Create a subset `repos_features` with relevant columns (`language`, `license_name`, `created_at`) and convert created_at into a new `year_created` column to capture when each repository was created.

- **Encoding Categorical Variables:** Convert categorical features (language, license_name, year_created) into dummy/indicator variables (0s and 1s). `drop_first=True` reduces multicollinearity by dropping one category per feature.

- **Defining Target Variable:** Set `target` to the binary high_engagement column, representing the outcome the model will predict.

- **Splitting Data into Training and Testing Sets:** Split data into training (70%) and testing (30%) sets. `random_state=42` ensures reproducibility.

- **Training the Logistic Regression Model:** Initialize and trains a logistic regression model with a maximum of `200` iterations, fitting it on the training data (`X_train`, `y_train`).

- **Extracting Top Features:** Create a DataFrame to store each feature and its impact (`Coefficient`) on predicting high engagement. `nlargest(3, 'Coefficient')` selects the top three most influential features.

- **Evaluating the Model:** Predict y_test using X_test and generates a report (`classification_report_data`) that shows model accuracy, precision, recall, and F1 score.

- **Displaying Top Features:** Uses `ace_tools` to display the top 3 predictive features for high engagement in a user-friendly format.


### Complete Code ###

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

engagement_threshold = 100

repositories_df['high_engagement'] = (repositories_df['stargazers_count'] > engagement_threshold).astype(int)

repos_features = repositories_df[['language', 'license_name', 'created_at']].copy()
repos_features['year_created'] = pd.to_datetime(repositories_df['created_at']).dt.year

repos_features = pd.get_dummies(repos_features[['language', 'license_name', 'year_created']], drop_first=True)

target = repositories_df['high_engagement']

X_train, X_test, y_train, y_test = train_test_split(repos_features, target, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': log_reg.coef_[0]})
top_features = coef_df.nlargest(3, 'Coefficient')

y_pred = log_reg.predict(X_test)
classification_report_data = classification_report(y_test, y_pred, output_dict=True)

import ace_tools as tools; tools.display_dataframe_to_user(name="Top 3 Engagement-Predictive Features", dataframe=top_features)

top_features, classification_report_data

```
### Result Table and Findings ###
Below are the top three features most associated with crossing the engagement threshold (100 stargazers):

| Rank | Feature          | Coefficient |
|------|------------------|-------------|
| 1    | MIT License      | 1.004537    |
| 2    | Go Language      | 0.314887    |
| 3    | JavaScript       | 0.218959    |


The model’s results reveal that license type and language choice are the strongest predictors of high engagement for repositories. The analysis suggests that the ***MIT license*** and specific languages like ***Go*** and ***JavaScript*** enhance repository visibility and popularity, especially among the Sydney GitHub community. This analysis underscores how license flexibility and language preference can impact repository engagement. By making informed choices about licensing and technology stack, developers can potentially attract more visibility and interaction for their projects.


