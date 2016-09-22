import pandas as pd
from urlparse import urlparse
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as numpy

class URLMapper( ):
    def __init__( self, csv_404s, csv_crawl, exclude, limit, threshold ):
        df = pd.read_csv( csv_404s )
        self.csv_404s = set( df[ "URL" ] )
        self.csv_crawl = pd.read_csv( csv_crawl, skiprows = 1 )
        self.texts_404 = set()
        self.texts_crawl = set()
        self.exclude = exclude
        self.limit = limit
        self.threshold = threshold
        self.no_match = set( )
        self.urlpairs = None

    def url2text( self, url ):
        u = urlparse( url ).path
        u = re.sub( "\..+", "", u )
        u = re.split( r"/|-|_", u )

        # eliminate single characters and digits
        t = [ x for x in u if len( x ) > 1 and re.match( r"^\d+$", x ) is None ]
        u = " ".join( t )
        return u
        # print re.sub("\..+", "", u)

    def pairwise_similarity( self, urls, texts ):
        # documents = [open(f) for f in text_files]
        tfidf = TfidfVectorizer( ).fit_transform( texts )
        # no need to normalize, since Vectorizer will return normalized tf-idf
        pairwise = tfidf * tfidf.T

        self.urlpairs = pd.DataFrame( pairwise.A, index = urls, columns = urls )

    def get_similar( self, url, texts_404, texts_crawl ):

        similar = None
        highest_value = 0

        for i, value in self.urlpairs[ url ].iteritems( ):
            if i in texts_crawl and value > highest_value:
                similar = i
                highest_value = value

                # print url, i, value

        return [similar, highest_value]

    def urlmatch( self ):

        urls = set( )
        texts = set( )
        texts_404 = set()
        texts_crawl = set()

        url_map = defaultdict( list )

        # process 404s first
        for i, url in enumerate( self.csv_404s ):

            if i < self.limit * 1000:

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

            if i < self.limit * 1000:

                text = self.url2text( url )

                if len( text ) > 0:
                    urls.add( text )
                    texts.add( text )
                    texts_crawl.add(text)
                else:
                    self.no_match.add( url )

        # initialize pairwise similarity matrix
        self.pairwise_similarity( urls, texts )

        # iterate urls to get similar
        for i, url in enumerate( self.csv_404s ):
            if i < self.limit:
                similar = None
                try:
                    text = self.url2text(url)
                    similar = self.get_similar( text, texts_404, texts_crawl )
                    if similar is not None:
                        if similar[1] > self.threshold:
                            url_map[ url ] = similar
                            # print url, similar
                        else:
                            self.no_match.add( url )
                except:
                    # print url, ' failed'
                    pass

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
args = parser.parse_args( [ "-u", "-x", "returnurl", "data/404s-all.csv", "data/response_codes_success_(2xx).csv" ] )

urlmapper = URLMapper( args.csv_404s, args.csv_crawl, args.exclude, args.limit, args.threshold )

url_map = []
if args.urlmatch:
    try:
        url_map = urlmapper.urlmatch( )
        pass
    except:
        pass
if args.h1match:
    url_map = urlmapper.h1match( )
if args.titlematch:
    url_map = urlmapper.titlematch( )

scores = []
matches = []
non_matches = []
for index, value in url_map.items():
    scores.append(value[1])
    if value[1] > 0.3:
        matches.append(value[0])
    else:
        non_matches.append(value[0])

print numpy.histogram(scores, 10, (0, 1))

print "Matches: ", len(matches)
print "Non-Matches: ", len(non_matches)

