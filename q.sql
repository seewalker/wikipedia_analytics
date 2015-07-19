-- why are the first many answers non-numerical? no google entries? yes, that is the problem. I guess I can put the not null thing in the where clause.
-- it took 2956180.009 milliseconds with both queries there.
SELECT wOut.title, (SELECT wIn.n :: float / sum(wOut.n) FROM wikithresh wIn WHERE (wIn.title = wOut.title) AND (wIn.referer = 'other-google'))
FROM wikiThresh wOut
GROUP BY wOut.title
ORDER BY (SELECT wIn.n :: float / sum(wOut.n) FROM wikithresh wIn WHERE (wIn.title = wOut.title) AND (wIn.referer = 'other-google')) DESC
