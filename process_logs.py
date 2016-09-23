import re
import glob
import pandas as pd
import ConfigParser
import argparse

parser = argparse.ArgumentParser( description = "process logs" )
parser.add_argument( '-p', '--project', type = str, help = 'project name' )

# test arguments
args = parser.parse_args([ "-p", "lehmans" ])
project = args.project

Config = ConfigParser.ConfigParser()
Config.read('data/' + project + '/config.ini')

pattern = Config.get('logs', 'pattern')
pattern = re.compile(pattern)

files = glob.glob("data/" + project + "/logs/*")

entries = list()

for _file in files:
    with open(_file) as f:
        for line in f:
            m=re.match(pattern, line)
            if m is not None:
                item = dict()
                (date, page, status) = m.groups()
                item["date"]=date
                item["page"]=page
                item["status"]=status
                if status == '404':
                    entries.append(item)

df = pd.DataFrame.from_dict(entries)

df= df.set_index("date")

print df.groupby("status").count()

df.to_csv("data/" + project + "/latest_404s.csv")

