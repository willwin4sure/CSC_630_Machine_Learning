import pandas as pd
from statistics import Counter

email_info = []

with open('emails.txt', 'r+') as f:
    for line in f:
        info = {}
        words = line.split(' ')
        try:
            words.remove('')
        except:
            pass
        info['Sender First Name'] = words[1]
        info['Sender Last Name'] = words[2]
        info['Sender Email'] = words[3][1:-1]
        info['Receiver First Name'] = words[5]
        info['Receiver Last Name'] = words[6]
        info['Receiver Email'] = words[7][1:-1]
        info['Day of the Week'] = words[9]
        info['Month'] = words[10]
        info['Date'] = words[11]
        info['Time'] = words[12]
        info['Year'] = words[13][0:-1]
        email_info.append(info)

days_of_week = {}
for info in email_info:
    if not info['Day of the Week'] in days_of_week.keys():
        days_of_week[info['Day of the Week']] = 1
    else:
        days_of_week[info['Day of the Week']] += 1

# print(max(days_of_week, key=days_of_week.get))

# domains = {}
domains = Counter()
for info in email_info:
    # domaina = info['Sender Email'][info['Sender Email'].index('@')+1:]
    domaina = info['Sender Email'].split('@')[1]
    # if not domaina in domains.keys():
    #     domains[domaina] = 1
    # else:
    #     domains[domaina] += 1
    domains[domaina] += 1
    # domainb = info['Receiver Email'][info['Receiver Email'].index('@')+1:]
    domainb = info['Receiver Email'].split('@')[1]
    # if not domainb in domains.keys():
    #     domains[domainb] = 1
    # else:
    #     domains[domainb] += 1
    domains[domainb] += 1

print(dict(domains))
print(min(domains, key=domains.get))

senders = Counter()
for info in email_info:
    sender = info['Sender Email']
    senders[sender] += 1

print(senders)
print(max(senders, key=senders.get))

df = pd.DataFrame.from_dict(email_info)
    
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
mapping = {day: i for i, day in enumerate(weekdays)}
key = df['Day of the Week'].map(mapping)
df = df.iloc[key.argsort()]

hist = df['Day of the Week'].hist()
fig = hist.get_figure()
fig.savefig("histogram.png")
# print(df.head(10))

df.to_csv('out.csv')
