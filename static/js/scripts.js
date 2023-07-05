let allResults = []; // variable to store all the results from the query [works when select all is clicked]
let allChecked = false; // to check whether select all is checked
let lastClickedItem = null;
let detailsDisplayed = false;

document.getElementById('search-form').addEventListener('submit', function(event) {
    event.preventDefault();
    document.getElementById('extract-form').style.display = 'block';
    const query = document.getElementById('query').value;
    fetch('/search', {
        method: 'POST',
        body: new FormData(event.target)
    })
    .then(response => response.json())
    .then(results => {
        allResults = results;
        const resultsList = document.getElementById('results-list');
        resultsList.innerHTML = '';

        
        // Create a "Check All" checkbox
        const checkAllCheckbox = document.createElement('input');
        checkAllCheckbox.type = 'checkbox';
        checkAllCheckbox.id = 'check-all-checkbox';
        

        // Create a label for the "Check All" checkbox
        const checkAllLabel = document.createElement('label');
        checkAllLabel.htmlFor = 'check-all-checkbox';
        checkAllLabel.textContent = 'Select All';
        checkAllLabel.style.paddingLeft = '5px'

        // Append the "Check All" checkbox and label to the results list
        resultsList.prepend(checkAllLabel);
        resultsList.prepend(checkAllCheckbox);
        

        // Create an array to store the individual checkboxes
        const checkboxes = [];

        const showResults = 10; // Number of results to initially show
        let visibleResults = Math.min(showResults, results.length); // Number of currently visible results, done in case results are less than 10
        const rowsPerResult = 3;

        results.slice(0, visibleResults).forEach(result => {
            const listItem = document.createElement('li');
            listItem.className = 'animate__animated animate__fadeInDown';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'selected_results';
            checkbox.value = JSON.stringify(result);

            // Append the checkbox to the list item
            listItem.appendChild(checkbox);

            // Create a container div in each result to contain info and metrics with display: flex
            const itemContainer = document.createElement('div');
            itemContainer.style.display = 'flex';
            itemContainer.style.alignItems = 'flex-start'; 
            itemContainer.style.justifyContent = 'space-between';

            const item = document.createElement('p');
            item.style.minWidth = '400px'; // Set a minimum width for the result info
            Object.entries(result).slice(0,rowsPerResult).forEach(([key, value]) => {
                item.innerHTML += `${key}: ${value}<br>`;
            });

            // Add a div for metric records
            const lastRecords = document.createElement('div');
            lastRecords.style.minWidth = '200px'; // Set a minimum width for the lastRecords
            lastRecords.style.flexGrow = '1'; // Add this line to make the div fill available space

            // Loop through last 2 records and append to the lastRecords div
            const p = document.createElement('p');
            Object.entries(result).slice(-3).forEach(([key, value]) => {
                p.innerHTML += `${key}: ${value}<br>`;
            });
            lastRecords.appendChild(p);

            // Append items to the container div
            itemContainer.appendChild(item);
            itemContainer.appendChild(lastRecords);

            listItem.appendChild(itemContainer)
            // On click, show all the details for that row on the side bar
            listItem.addEventListener('click', function() {
                if(lastClickedItem) {
                    lastClickedItem.style.backgroundColor = "#ffffff";
                }
                listItem.style.backgroundColor ="#efefef";
                lastClickedItem = listItem;
                if(!detailsDisplayed) {
                    document.getElementById('details-sidebar').style.display = 'block';
                }
                const detailsContent = document.getElementById('details-content');
                detailsContent.innerHTML = '';
                

                // Retrieve image for this row
                fetch('/retrieve_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                })
                .then((response) => response.json())
                .then((data) => {
                    // Create an img element and set its src property to the base64-encoded string
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + data.image_base64;
                    img.className = 'animate__animated animate__fadeInRight';
                    img.style.display = 'block';
                    img.style.margin = '0 auto';
                    img.style.maxWidth = '40%';
        
                    // Insert the img element at the beginning of the detailsContent div
                    detailsContent.insertBefore(img, detailsContent.firstChild);
                });

                // list out all the details for this row on the sidebar
                Object.entries(result).forEach(([key, value]) => {
                    const detail = document.createElement('p');
                    detail.className = 'animate__animated animate__fadeInRight';
                    detail.textContent = `${key}: ${value}`;
                    detailsContent.appendChild(detail);
                });
            });

            resultsList.appendChild(listItem);

            // Add the checkbox to the checkboxes array
            checkboxes.push(checkbox);
        });

        // Most of the code below is a repetition to the above code
        // Load more button event listener
        const loadMoreBtn = document.createElement('button');
        loadMoreBtn.textContent = 'Load More';
        loadMoreBtn.className = 'btn btn-primary';
        resultsList.appendChild(loadMoreBtn);

        loadMoreBtn.addEventListener('click', function() {
            const remainingResults = results.length - visibleResults;
            const loadCount = Math.min(showResults, remainingResults); // Calculate the number of results to load

            results.slice(visibleResults, visibleResults + loadCount).forEach(result => {
                const listItem = document.createElement('li');
                listItem.className = 'animate__animated animate__fadeInDown';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'selected_results';
                checkbox.value = JSON.stringify(result);
                checkbox.checked = checkAllCheckbox.checked;

                // Append the checkbox to the list item
                listItem.appendChild(checkbox);

                const itemContainer = document.createElement('div');
                itemContainer.style.display = 'flex';
                itemContainer.style.alignItems = 'flex-start'; 
                itemContainer.style.justifyContent = 'space-between';

                const item = document.createElement('p');
                item.style.minWidth = '400px'; // Set a minimum width for the item
                Object.entries(result).slice(0,rowsPerResult).forEach(([key, value]) => {
                    item.innerHTML += `${key}: ${value}<br>`;
                });

                // Add a div for last records
                const lastRecords = document.createElement('div');
                lastRecords.style.minWidth = '200px'; // Set a minimum width for the lastRecords
                lastRecords.style.flexGrow = '1'; // Add this line to make the div fill available space

                // Loop through last 3 records and append to the lastRecords div
                const p = document.createElement('p');
                Object.entries(result).slice(-3).forEach(([key, value]) => {
                    p.innerHTML += `${key}: ${value}<br>`;
                });
                lastRecords.appendChild(p);

                // Append items to the container div
                itemContainer.appendChild(item);
                itemContainer.appendChild(lastRecords);

                listItem.appendChild(itemContainer)
                
                listItem.addEventListener('click', function() {
                    if(lastClickedItem) {
                        lastClickedItem.style.backgroundColor = "#ffffff";
                    }
                    listItem.style.backgroundColor ="#efefef";
                    lastClickedItem = listItem;
                    const detailsContent = document.getElementById('details-content');
                    detailsContent.innerHTML = '';
                    // Send the data to the server
                    fetch('/retrieve_image', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    })
                    .then((response) => response.json())
                    .then((data) => {
                        // Create an img element and set its src property to the base64-encoded string
                        const img = document.createElement('img');
                        img.src = 'data:image/png;base64,' + data.image_base64;
                        img.className = 'animate__animated animate__fadeInRight';
                        img.style.display = 'block';
                        img.style.margin = '0 auto';
                        img.style.maxWidth = '40%';
            
                        // Insert the img element at the beginning of the detailsContent div
                        detailsContent.insertBefore(img, detailsContent.firstChild);
                    });
                    
                    Object.entries(result).forEach(([key, value]) => {
                        const detail = document.createElement('p');
                        detail.className = 'animate__animated animate__fadeInRight';
                        detail.textContent = `${key}: ${value}`;
                        detailsContent.appendChild(detail);
                    });
                });
                resultsList.appendChild(listItem);

                // Add the checkbox to the checkboxes array
                checkboxes.push(checkbox);

                resultsList.appendChild(loadMoreBtn);
            });

            visibleResults += loadCount;

            if (visibleResults >= results.length) {
                loadMoreBtn.style.display = 'none'; // Hide the "Load More" button if all results are loaded
            }
        });  
        
        // Add an event listener to the "Check All" checkbox
        document.getElementById('check-all-checkbox').addEventListener('change', function() {
            // stor this in the global variable
            allChecked = checkAllCheckbox.checked;
            // Set the checked property of each checkbox based on the "Check All" checkbox state
            checkboxes.forEach(checkbox => {
                checkbox.checked = checkAllCheckbox.checked;
            });
        });
    });
});

document.getElementById('extract-form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Show the loading overlay
    const loadingOverlay = document.createElement('div');
    loadingOverlay.id = 'loading-overlay';
    const loadingSpinner = document.createElement('div');
    loadingSpinner.id = 'loading-spinner';

    loadingOverlay.appendChild(loadingSpinner);
    document.body.appendChild(loadingOverlay);

    // Disable the button
    const extractButton = document.getElementById('extract-button');
    extractButton.disabled = true;
        
    // Get the selected checkboxes
    const checkboxes = document.querySelectorAll('input[name="selected_results"]:checked');
    
    // Create an array to store the selected results
    let selectedResults = [];
    
    // Iterate over the selected checkboxes and add their values to the array
    checkboxes.forEach(checkbox => {
        const result = JSON.parse(checkbox.value);
        selectedResults.push(result);
    });
    
    // Get the form inputs for start_date and end_date
    const startDate = document.getElementById('start_date').value;
    const endDate = document.getElementById('end_date').value;
    const table = document.getElementById('table').value;    
    const extraction = document.getElementById('extraction').value; 
    if(allChecked) {
        selectedResults = allResults;
    }

    // Create the payload object to send to the server
    const payload = {
        start_date: startDate,
        end_date: endDate,
        table: table,
        extraction: extraction,
        results: selectedResults
    };
    

    // Send the data to the server
    fetch('/extract', {
        method: 'POST',
        body: JSON.stringify(payload),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.blob())  // get a Blob (binary large object) from the response
    .then(blob => {
        // create a link element
        const link = document.createElement('a');
        
        // create a URL for the Blob
        link.href = URL.createObjectURL(blob);
        
        // specify that this link is for a download
        link.download = 'data.csv';  // replace 'file.csv' with the actual filename
       
        
        document.body.appendChild(link);
        link.click();

        document.body.removeChild(link);
        document.body.removeChild(loadingOverlay);
        extractButton.disabled = false;
    });
});







