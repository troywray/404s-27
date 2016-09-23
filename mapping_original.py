import pandas as pd
from urlparse import urlparse
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as numpy

class URLMapper():
    def __init__(self, project, master_map, latest_crawl, latest_404s):
        df = pd.read_csv( "data/" + project + "/" + latest_404s )
        self.csv_404s = set( df[ "URL" ] )
        self.csv_crawl = pd.read_csv( "data/" + project + "/" + latest_crawl, skiprows = 1 )
        self.master_map = pd.read_csv( "data/" + project + "/" + master_map )
        self.match = ''
        self.force = ''
        self.texts_404 = set()
        self.texts_crawl = set()
        self.exclude = []
        self.limit = 10000
        self.threshold = 0.2
        self.no_match = set( )
        self.urlpairs = None
        self.text_url_map = {}
        self.csv_404s_meta = {}


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
        # print pairwise
        # print pairwise.A
        self.urlpairs = pd.DataFrame( pairwise.A, index = urls, columns = texts )

    def get_similar( self, url, urls_crawl, texts_crawl ):
        """
        scans all url pairs and finds the highest score that's also in texts_crawl
        :param url:         The tokenized text to search for
        :param urls_crawl:  valid urls
        :param texts_crawl: tokenized texts of valid urls
        :return:            match and score
        """
        similar = None
        highest_value = 0

        for i, value in self.urlpairs[ url ].iteritems( ):
            if i in texts_crawl and value > highest_value:
                similar = i
                highest_value = value

        return [similar, highest_value]

    def urlmatch( self ):

        urls = set( )
        texts = set( )
        urls_404 = set()
        urls_crawl = set()
        texts_404 = set()
        texts_crawl = set()

        url_map = defaultdict( list )

        # process 404s first
        for i, url in enumerate( self.csv_404s ):
            # only deal with match urls
            if self.match != '':
                if url.find(self.match) == -1:
                    continue

            if i < self.limit * 1000:
                path = urlparse(url).path

                text = self.url2text( path )
                self.csv_404s_meta[path] = [text, '', '', 0, False]  # tokens, matched tokens, url, score, matched
                if len( text ) > 0:
                    urls.add( text )
                    texts.add( text )
                    urls_404.add(path)
                    texts_404.add(text)
                    pass
                else:
                    self.no_match.add( url )
        print len(urls_404), " 404s added"

        # process crawled urls
        for i, url in enumerate( self.csv_crawl[ "Address" ] ):
            # ignore any specified urls
            for exclude in self.exclude:
                if url.find( exclude ) > 0:
                    continue

            # ignore urls unless they include force string
            if self.force != '':
                if url.find(self.force) == -1:
                    continue

            path = urlparse(url).path
            if path.find('/product/') != -1:
                s = path.split('/')
                if len(s) > 1:
                    s = s[0:len(s)-1]
                path = "/".join(s)

            text = self.url2text(path)

            if len( text ) > 0:
                urls.add( text )
                texts.add( text )
                urls_crawl.add(path)
                texts_crawl.add(text)
                if self.text_url_map.has_key(text):
                    if False:
                        print "duplicate text from crawl: "
                        print "text was: ", text
                        print "this url: ", url
                        print "previous url: ", self.text_url_map[text]
                        # exit(1)
                self.text_url_map[text] = path
            else:
                self.no_match.add( path )

        print len(urls_crawl), " crawled urls added"
        # initialize pairwise similarity matrix
        self.pairwise_similarity( urls, texts )

        # iterate urls to get similar
        for i, url in enumerate( self.csv_404s_meta ):
            path = url
            if i < self.limit:
                similar = None
                text = self.url2text(path)
                similar = None
                try:
                    similar = self.get_similar( text, urls_crawl, texts_crawl )
                except:
                    pass
                if similar is not None:
                    if True or similar[1] > self.threshold:
                        url_map[ path ] = similar
                        # print url, similar
                        match_url = ''
                        try:
                            match_url = self.text_url_map[ similar[ 0 ] ]
                        except:
                            pass
                        surl = [text, similar[0], similar[1], match_url, True]
                        self.csv_404s_meta[path] = surl
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
import ConfigParser
import json

parser = argparse.ArgumentParser( description = "map 404 urls" )
parser.add_argument( '-u', '--urlmatch', action = 'store_true', help = 'match using urls' )
parser.add_argument( '-e', '--h1match', action = 'store_true', help = 'match using h1s' )
parser.add_argument( '-t', '--titlematch', action = 'store_true', help = 'match using titles' )
parser.add_argument( '-l', '--limit', type = int, default = 10000, help = 'process maximun urls' )
parser.add_argument( '-o', '--threshold', type = int, default = 0.1, help = 'minimum similarity score' )
# parser.add_argument( '-x', '--exclude', type = str, help = 'exclude urls with this pattern' )
parser.add_argument( '-p', '--project', type = str, help = 'match using h1s' )
parser.add_argument( "master_map", help = "master map" )
parser.add_argument( "csv_404s", help = "csv file with 404s" )
parser.add_argument( "csv_crawl", help = "csv file with website crawl" )


# test arguments
args = parser.parse_args( [ "-u", "-p", "lehmans", "master_map.csv", "latest_404s.csv", "response_codes_success_(2xx).csv" ] )

project = args.project
Config = ConfigParser.ConfigParser()
Config.read('data/' + project + '/config.ini')

force_match = json.loads(Config.get('force_patterns', 'match'))
force = json.loads(Config.get('force_patterns', 'force'))

for i, match in enumerate(force_match):
    print "processing ", match
    urlmapper = URLMapper( project, args.master_map, args.csv_crawl, args.csv_404s )

    exclude = json.loads( Config.get( '404s', 'exclude' ) )
    urlmapper.exclude = exclude

    urlmapper.match = match
    urlmapper.force = force[i]

    url_map = urlmapper.urlmatch()

    scores = [ ]
    matches = [ ]
    non_matches = [ ]
    for index, value in url_map.items( ):
        scores.append( value[ 1 ] )
        if value[ 1 ] > 0.3:
            matches.append( value[ 0 ] )
        else:
            non_matches.append( value[ 0 ] )

    print numpy.histogram( scores, 10, (0, 1) )

    print "Matches: ", len( matches )
    print "Non-Matches: ", len( non_matches )

    output = ''
    for key, entry in urlmapper.csv_404s_meta.items():
        # print key, ", ", str(entry[0]), ", ", str(entry[1]), ", ", str(entry[2]), ", ", str(entry[3]), ", ", str(entry[4])
        output += str(key) + ", " + str(entry[0]) + ", " + str(entry[1]) + ", " + str(entry[2]) + ", " + str(entry[3]) + ", " + str(entry[4]) + "\r"

    filename = "data/" + project + "/new_mappings." + match.replace('/', '__') + ".csv"

    handle = open(filename, "w")
    handle.write(output)
    handle.close()
    print
    # df = pd.DataFrame.from_dict( urlmapper.csv_404s_meta.items() )
    # df.to_csv( filename )

exit()





url_map = []
if args.urlmatch:
    url_map = urlmapper.urlmatch( )
    pass
    pass
if args.h1match:
    url_map = urlmapper.h1match( )
if args.titlematch:
    url_map = urlmapper.titlematch( )


# print urlmapper.urlpairs

for m in matches:
    print m, urlmapper.text_url_map[m]



for key, entry in urlmapper.csv_404s_meta.items():
    print key, ", ", entry[0], ", ", entry[1], ", ", entry[2], ", ", entry[3], ", ", entry[4]