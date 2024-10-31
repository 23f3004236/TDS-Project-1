**README.md** begin with 3 bullet points. Each bullet is no more than 50 words.

- ### An explanation of how I scraped the data: ###
The code uses `GitHub API` and `Personal GitHub Token` to scrape all the user profiles from Sydney with over 100 followers. It fetches user details (e.g., login, company, bio) and 500 most recently pushed repositories, handling rate limits with `time.sleep()`. Data is saved into `users.csv` and `repositories.csv` after organizing and cleaning fields like company names and licenses.
   
- ### The most interesting and surprising fact I found after analyzing the the data: ###
My analysis unveils a fascinating trend that Sydney developers’ repositories with the **MIT license** and languages like **Go** and **JavaScript** attract the highest engagement. This insight reveals how flexible licensing and strategic language choices can boost visibility and popularity, making these factors powerful tools for attracting more interaction.
   
- ### An actionable recommendation for developers based on my analysis: ###
To maximize visibility and engagement, developers should license projects under MIT for its openness and consider coding in high-demand languages like Go and JavaScript. This combination not only attracts more contributors but also increases project appeal in the broader developer community.
