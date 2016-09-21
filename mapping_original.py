import requests
import urllib
import pandas as pd
import json
from urlparse import urlparse
import re
from bs4 import BeautifulSoup

urls=["https://www.lehmans.com/c-465-stocking-stuffers.aspx","https://www.lehmans.com/p-1230.aspx","http://www.gumps.com/p/tommy-valentine-woodcut-puzzle", "https://www.lehmans.com/p-1613-electric-cream-separator","https://www.lehmans.com/product/manual-cream-separator"]
#matchType=prefix|host|domain
#&collapse=digest|urlkey

def url2text(url):
    u = urlparse(url).path
    u = re.sub("\..+", "", u)
    u = re.split(r"/|-|_", u)

    #eliminate single characters and digits
    t = [x for x in u if len(x) > 1 and re.match(r"^\d+$", x) is None]
    u = " ".join(t)
    return u
    #print re.sub("\..+", "", u)

#re.match(r"^\d+$", "465")

def get_cdx_hits( u ):
    cdx = "http://web.archive.org/cdx/search/cdx?url={}&output=json&limit=3&matchType=exact&collapse=digest".format(
        urllib.quote_plus( u ) )
    r = requests.get( cdx )
    if r:
        print r.headers[ 'content-type' ]

        df = pd.read_json( r.text )
        df.columns = df.iloc[ 0 ]
        df = df.reindex( df.index.drop( 0 ) )
        df.set_index( "timestamp" )

        return df

    return None

# print url2text(url)

df=get_cdx_hits("https://www.lehmans.com/p-1230.aspx")
print df


def get_latest_cache( hits ):
    wayback = "https://web.archive.org/web/{timestamp}/{original}"  # .format(timestamp, original)

    for index, row in hits.iterrows( ):
        statuscode = row[ "statuscode" ]

        if statuscode == "200":
            url_cache = wayback.format( original = row[ "original" ], timestamp = row[ "timestamp" ] )

            r = requests.get( url_cache )
            if r:
                print r.headers[ 'content-type' ]
                content = r.content
                with open( "cache.html", "w+" ) as f:
                    f.write( content )
                    print "saved cache.html"
                return (200, content)
    # no 200
    print "no 200"
    if statuscode == "301":
        # need to refetch to get redirect url
        url_cache = wayback.format( original = row[ "original" ], timestamp = row[ "timestamp" ] )

        r = requests.get( url_cache )
        if r:
            print r.headers[ 'content-type' ]
            content = r.content
            with open( "cache.html", "w+" ) as f:
                f.write( content )
                print "saved cache.html"
            return (301, content)

def get_text(content):
    soup = BeautifulSoup(content, "lxml")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    #text = [chunk for chunk in chunks if chunk]

    return text


def get_h1( content ):
    soup = BeautifulSoup( content, "lxml" )
    # get main page topic
    h1 = soup.find( 'h1' )

    if h1:
        return h1.get_text( )
        # soup.find('h1',attrs={'itemprop':'name'})


def get_redirect( content ):
    soup = BeautifulSoup( content, "lxml" )
    # get main page topic
    p = soup.findAll( 'p', attrs = { 'class':'code shift target' } )

    # print p
    if len( p ) > 1:
        # second node has the redirect
        return p[ 1 ].get_text( )

    print "no redirect"

#print get_text(content_cache)
#print get_h1(content_cache)
# print get_redirect(content)

# del content_cache

hits=get_cdx_hits("https://www.lehmans.com/p-1230.aspx")
print hits

code,content=get_latest_cache(hits)
if code == "200":
    h1 = get_h1(content)
    print h1


def check( url ):
    text = url2text( url )

    if len( text ) > 1:
        print text
        return text
    else:
        print "checking page in wayback machine"
        hits = get_cdx_hits( url )
        # print hits
        code, content = get_latest_cache( hits )

        if code == 200:
            h1 = get_h1( content )
            print h1
            return h1

        if code == 301:
            print code

            redirect = get_redirect( content )
            print redirect

            # try again
            return check( redirect )

data = list()

for i,x in enumerate(urls):
    print x
    result = check(x)

    if len(result)>0:
        #data[i] = result
        data.append(result)

print data

# from sklearn.feature_extraction.text import TfidfVectorizer
# vec = TfidfVectorizer()
# D = vec.fit_transform(text)
# voc = dict((i, w) for w, i in vec.vocabulary_.items())

#https://github.com/leonsim/simhash/pull/11/files

# print voc

#import re
#from simhash import Simhash
#def get_features(s):
#    width = 3
#    s = s.lower()
#    s = re.sub(r'[^\w]+', '', s)
#    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]

#print '%x' % Simhash(get_features('How are you? I am fine. Thanks.')).value
#print '%x' % Simhash(get_features('How are u? I am fine.     Thanks.')).value
#print '%x' % Simhash(get_features('How r you?I    am fine. Thanks.')).value

#get_features('How are you? I am fine. Thanks.')

#print Simhash(data[3]).distance(Simhash(data[4]))
#print Simhash(data[3]).distance(Simhash(data[3]))
#print Simhash(data[0]).distance(Simhash(data[3]))
#print Simhash(data[2]).distance(Simhash(data[3]))
#print Simhash(data[1]).distance(Simhash(data[3]))

#import re
#from simhash import Simhash, SimhashIndex
#def get_features(s):
#    width = len(data)#3
#    s = s.lower()
#    s = re.sub(r'[^\w]+', '', s)
#    return [s[i:i + width] for i in range(max(len(s) - width + 1, 1))]


#data = {
#    1: u'How are you? I Am fine. blar blar blar blar blar Thanks.',
#    2: u'How are you i am fine. blar blar blar blar blar than',
#    3: u'This is simhash test.',
#}
#objs = [(str(k), Simhash(get_features(v))) for k, v in data.items()]
#index = SimhashIndex(objs, k=3)
#index = SimhashIndex(objs, k=len(data))

#print index.bucket_size()

#s1 = Simhash(get_features(u'electric cream separator'))
#print index.get_near_dups(s1)

from sklearn.feature_extraction.text import TfidfVectorizer

# print pairwise_similarity.A

# label rows and columns
# df_sim=pd.DataFrame(pairwise_similarity.A, index=urls,columns=urls)
# print df_sim

def pairwise_similarity( urls, texts ):
    # documents = [open(f) for f in text_files]
    tfidf = TfidfVectorizer( ).fit_transform( texts )
    # no need to normalize, since Vectorizer will return normalized tf-idf
    pairwise = tfidf * tfidf.T

    return pd.DataFrame( pairwise.A, index = urls, columns = urls )

# df_sim=pairwise_similarity(urls,data)

def get_similar( url, df ):
    similar = None
    highest_value = 0

    for i, value in df.loc[ url ].iteritems( ):
        if i != url and value > highest_value:
            similar.append( (i, value) )

    return similar

# print get_similar("https://www.lehmans.com/p-1613-electric-cream-separator", df_sim)

# import requests
# import urllib
# import pandas as pd
import json
from urlparse import urlparse
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

import random

class URLMapper( ):
    def __init__( self, csv_404s, csv_crawl, exclude, limit, threshold ):
        df = pd.read_csv( csv_404s )
        self.csv_404s = set( df[ "URL" ] )
        self.csv_crawl = pd.read_csv( csv_crawl, skiprows = 1 )
        self.csv_internal = pd.read_csv( 'data/lehmans_internal_html.csv', skiprows = 1 )
        self.exclude = exclude
        self.limit = limit
        self.threshold = threshold
        self.no_match = set( )
        self.urlpairs = None
        self.word_frequency_table = {}

    def randomWord(self):
        WORDS = ("apple", "orange", "banana", "kiwi", "plum", "blackberry", "cherry", "melon", "grape", "pear")
        word = random.choice( WORDS )
        return word

    def addToWordFrequencyTable(self, phrase):
        words = phrase.split()
        for word in words:
            if not word in self.word_frequency_table:
                self.word_frequency_table[word] = 1
            else:
                self.word_frequency_table[word] += 1

    def url2text( self, url ):
        u = urlparse( url ).path
        u = re.sub( "\..+", "", u )
        u = re.split( r"/|-|_", u )

        # eliminate single characters and digits @todo
        t1 = [ x for x in u if len( x ) > 2 and re.match( r"^\d+$", x ) is None ]
        t2 = [ x for x in u if len( x ) > 2 and re.match( r"^\d+$", x ) is None ]
        # t = [ x for x in u if len( x ) > 4 ]
        u = " ".join( t1 )
        # u = u.replace('product', self.randomWord())
        # u = u.replace('book', self.randomWord())
        # u = u.replace('green', self.randomWord())
        # u = u.replace('dietz', 'dietz dietz dietz')
        # u = u.replace('green', 'green green green green')
        self.addToWordFrequencyTable(u)
        return u
        # print re.sub("\..+", "", u)

    def pairwise_similarity( self, urls, texts ):
        # documents = [open(f) for f in text_files]
        tfidf = TfidfVectorizer( ).fit_transform( texts )
        # no need to normalize, since Vectorizer will return normalized tf-idf
        pairwise = tfidf * tfidf.T

        self.urlpairs = pd.DataFrame( pairwise.A, index = urls, columns = texts )

    def get_similar( self, url, texts_404, texts_crawl ):

        similar = None
        highest_value = 0

        for i, value in self.urlpairs[ url ].iteritems( ):
            if i not in self.csv_404s:
                if value > 10.05:
                    print url, "---", i, value
                if i in texts_crawl and value > highest_value:
                    similar = i
                    highest_value = value
                    # print url, "***", i, value

        print url, '***', similar, highest_value
        return similar

    def urlmatch( self ):

        urls = set( )
        texts = set( )
        texts_404 = set()
        texts_crawl = set()

        url_map = defaultdict( list )

        # process 404s first
        for i, url in enumerate( self.csv_404s ):

            if i < self.limit:

                text = self.url2text( url )

                if len( text ) > 0:
                    urls.add( text )
                    texts.add( text )
                    texts_404.add(text)
                    pass
                else:
                    self.no_match.add( url )

        # process crawled urls
        for i, url in enumerate( self.csv_crawl[ "Address" ] ):

            if url.find( self.exclude ) > 0:
                continue

            if i < self.limit:

                text = self.url2text( url )

                if len( text ) > 0:
                    if text.lower().find('wick') != -1 and text.lower().find('wick') != -1:
                        print "Adding ", text
                    urls.add( text )
                    texts.add( text )
                    texts_crawl.add(text)
                else:
                    self.no_match.add( url )

        if False:
            # process lehmans_internal_html.csv
            for i, url in enumerate( self.csv_internal[ "Address" ] ):

                if url.find( self.exclude ) > 0:
                    continue

                if i < self.limit:

                    text = self.url2text( url )

                    if len( text ) > 0:
                        urls.add( text )
                        texts.add( text )
                        texts_crawl.add(text)
                    else:
                        self.no_match.add( url )

        # initialize pairwise similarity matrix
        self.pairwise_similarity( urls, texts )
        # sorted_word_frequency = sorted(self.word_frequency_table.items(), key=lambda x: (-x[1], x[0]))
        # sorted_words = sorted(self.word_frequency_table.items())

        # iterate urls to get similar
        for url in self.csv_404s:
        # for url in ['flat wick pack for oil lamps', ]:
            text = self.url2text( url )
            similar = None
            try:
                similar = self.get_similar( text, texts_404, texts_crawl )
            except:
                pass
            if similar is not None:
                if len( similar ) > self.threshold:
                    url_map[ url ] = similar
                    # print url, similar
                else:
                    self.no_match.add( url )

        return url_map

        # TODO save url_map to csv

    def h1match( self ):
        pass

    def titlematch( self ):
        pass

import argparse
import sys

parser = argparse.ArgumentParser( description = "map 404 urls" )
parser.add_argument( '-u', '--urlmatch', action = 'store_true', help = 'match using urls' )
parser.add_argument( '-e', '--h1match', action = 'store_true', help = 'match using h1s' )
parser.add_argument( '-t', '--titlematch', action = 'store_true', help = 'match using titles' )
parser.add_argument( '-l', '--limit', type = int, default = 10000, help = 'process maximun urls' )
parser.add_argument( '-o', '--threshold', type = int, default = 0.1, help = 'minimum similarity score' )
parser.add_argument( '-x', '--exclude', type = str, help = 'exclude urls with this pattern' )
parser.add_argument( "csv_404s", help = "csv file with 404s" )
parser.add_argument( "csv_crawl", help = "csv file with website crawl" )

# test arguments
args = parser.parse_args( [ "-u", "-x", "returnurl", "data/lehmans-404s.csv", "data/response_codes_success_(2xx).csv" ] )

urlmapper = URLMapper( args.csv_404s, args.csv_crawl, args.exclude, args.limit, args.threshold )

if args.urlmatch:
    url_map = urlmapper.urlmatch( )
    pass
if args.h1match:
    url_map = urlmapper.h1match( )
if args.titlematch:
    url_map = urlmapper.titlematch( )
