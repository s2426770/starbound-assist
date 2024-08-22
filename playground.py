import requests  
from bs4 import BeautifulSoup  

def count_a_tags(url):  
    # Fetch the HTML content from the URL  
    response = requests.get(url)  
    html_content = response.content  

    # Parse the HTML using BeautifulSoup  
    soup = BeautifulSoup(html_content, 'html.parser')  

    # Find all the <a> tags  
    a_tags = soup.find_all('img')  

    # Return the count of <a> tags  
    return len(a_tags)  

# Example usage  
url = "https://starbounder.org/Starbound_Wiki"  
a_tag_count = count_a_tags(url)  
print(f"The number of <a> tags in the URL '{url}' is: {a_tag_count}")