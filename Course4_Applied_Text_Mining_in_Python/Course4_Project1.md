# Project 1

In this project, we worked with messy medical data and using regex to extract relevant infromation from the data. 

Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.

The goal of this project is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 

Here is a list of some of the variants that might be found in this dataset:
* 04/20/2009; 04/20/09; 4/20/09; 4/3/09
* Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
* 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
* Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
* Feb 2009; Sep 2009; Oct 2010
* 6/2008; 12/2009
* 2009; 2010

Once these date patterns have been extracted from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
* Assume all dates in xx/xx/xx format are mm/dd/yy
* Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
* If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
* If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
* Watch out for potential typos as this is a raw, real-life derived dataset.

With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.

For example if the original series was this:

    0    1999
    1    2010
    2    1978
    3    2015
    4    1985

Function should return this:

    0    2
    1    4
    2    0
    3    1
    4    3

Score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.

*Return a Series of length 500 and dtype int.*


```python
import pandas as pd
import re

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
```

## Student work below

```python
from datetime import datetime

# matches: 10/3/98; 4/25/1997; etc.
date_regex1 = r'(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{2,4})'

# matches: 24 January, 2002; 13 Feb 2005; etc.
date_regex2 = r'(?P<day>\d{1,2} )(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\.?,? (?P<year>\d{4})'

# matches: January 24, 2002; Feb 24 2005; etc.
date_regex3 = r'(?P<month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\.?\s?(?P<day>\d{1,2})?,? (?P<year>\d{4})'

# matches: 6/1998; 2005
date_regex4 = r'(?P<month>\d{1,2})?[-/]?(?P<year>\d{4})'

# just check month[:3] against keys
months = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

def date_sorter():

    matches = []
    non_matches = []

    for sentence in df:
        m1 = re.search(date_regex1, sentence)
        m2 = re.search(date_regex2, sentence)
        m3 = re.search(date_regex3, sentence)
        m4 = re.search(date_regex4, sentence)

        if m1:
            day = m1.group('day')
            month = m1.group('month')
            year = m1.group('year')
            if len(day) == 1:
                day = '0' + day
            if len(month) == 1:
                month = '0' + month
            if len(year) == 2:
                year = '19' + year
            matches.append([month, day, year])

        elif m2:
            if m2.group('day'):
                day = m2.group('day')
            else:
                day = '01'
            month = months[m2.group('month')[:3]]
            year = m2.group('year')
            if len(year) == 2:
                year = '19' + year
            matches.append([month, day, year])
            
        elif m3:
            if m3.group('day'):
                day = m3.group('day')
            else:
                day = '01'
            month = months[m3.group('month')[:3]]
            year = m3.group('year')
            if len(year) == 2:
                year = '19' + year
            matches.append([month, day, year])

        elif m4:
            day = '01'
            if m4.group('month'):
                month = m4.group('month')
                if len(month) == 1:
                    month = '0' + month
            else:
                month = '01'
            year = m4.group('year')
            matches.append([month, day, year])

        else:
            non_matches.append(sentence)
            
    # date components in list format
    dates_raw = pd.Series(matches)
    
    # formats list of date components
    format_dates = lambda d: datetime.strptime(" ".join(d), '%m %d %Y')
    
    # apply formatting to dates
    dates_formatted = dates_raw.apply(format_dates)

    # convert dates to datetime objects
    dates_converted = pd.to_datetime(dates_formatted)
    
    return dates_converted.argsort()
