import requests

r = requests.get('https://en.wikipedia.org/wiki/Phillips_Academy')
print(list(filter(lambda x: not x.startswith('_'), dir(r))))
print(r.url)
# print(r.status_code)
# print(r.text)
# print(r.content)

text = r.text
links = []

while 'href="' in text:
    i = text.index('href="')+6
    link = ""
    while text[i] != '"':
        link += text[i]
        i += 1
    links.append(link)
    text = text[i+1:]

wiki_count = 0
non_wiki_count = 0
for link in links:
    if (link.startswith('/') and not link.startswith('//')) or link.startswith('#') or 'wiki' in link:
        wiki_count += 1
    else:
        non_wiki_count += 1
    
print(links)
print(wiki_count, non_wiki_count)
