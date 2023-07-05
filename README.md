1. app.py: The main entry point for the Flask application.
2. templates/index.html: The main HTML template for the frontend.
3. static/css/styles.css: The CSS file for styling the frontend.
4. static/js/scripts.js: The JavaScript file for handling user interactions and AJAX requests.
5. requirements.txt: The Python dependencies for the Flask application.

Dependencies
"animate.css": "^4.1.1",
"bootstrap": "^5.3.0",
"font-awesome": "^6.4.0"
{
All these 3 dependencies are available in the folder. However, this is an explanation on how to re-install them if wanted.
You need to download all.min.css from font-awesome version 6.4.0 and other files such as svgs etc. and copy it to css folder
download animate.min.css from animate.css and copy it along with the respective files that are downloaded to css, download bootstrap.min.css from bootstrap and copy it along with the respective files that are downloaded to css. 
If some things are not showing check the version and change the commands based on their website instructions

You can install animate.css and bootstrap with npm install animate.css bootstrap
However, font-awesome needs to be installed manually
}

To run the application type: python app.py [data.csv] [fit]
where data.csv is the csv file name and fit is 0 or 1 depending on whether you want to load the previous pre-processed data for this file or fit it again

or you can use the activate.bat/run.sh



How to use the application?

Use the search bar to search for a person based on any of the traits that show in the details section.

What are these metrics?
Similarity shows how similar your query is to the full record.
Any Match is true when any of the keywords written match a part of the record.
Full Match is true when all the keywords written match a record.

Extraction:
1- Choose the date range or leave empty for all dates
2- Choose the system to extract from
3- Choose the extraction type
 3.1 Extract Selection will extract the items checked
 3.2 Extract Any Matches will extract items that have Any Match as true
 3.3 Extract Full Matches will extract items that have Full Match as true