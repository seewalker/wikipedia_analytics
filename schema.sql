--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

--
-- Name: logt; Type: DOMAIN; Schema: public; Owner: akseewa11
--

CREATE DOMAIN logt AS text
	CONSTRAINT logt_check CHECK ((((VALUE ~ 'link'::text) OR (VALUE ~ 'redlink'::text)) OR (VALUE ~ 'other'::text)));


ALTER DOMAIN public.logt OWNER TO akseewa11;

--
-- Name: wiki_thresh(integer); Type: FUNCTION; Schema: public; Owner: akseewa11
--

CREATE FUNCTION wiki_thresh(thresh integer) RETURNS TABLE(referer_id integer, id integer, n integer, referer character varying, title character varying, type logt)
    LANGUAGE sql
    AS $$SELECT * FROM WikiLog WHERE title IN (SELECT title FROM WikiLog GROUP BY title HAVING sum(n) > thresh)$$;


ALTER FUNCTION public.wiki_thresh(thresh integer) OWNER TO akseewa11;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: wikilog; Type: TABLE; Schema: public; Owner: akseewa11; Tablespace: 
--

CREATE TABLE wikilog (
    referer_id integer,
    id integer,
    n integer NOT NULL,
    referer character varying NOT NULL,
    title character varying NOT NULL,
    type logt NOT NULL,
    CONSTRAINT wikilog_id_check CHECK ((id >= 0)),
    CONSTRAINT wikilog_n_check CHECK ((n >= 0)),
    CONSTRAINT wikilog_referer_id_check CHECK ((referer_id >= 0))
);

-- Filter out pages visited less than 300 times.
CREATE TABLE wikiThresh AS
(SELECT * FROM WikiLog WHERE title IN (SELECT title FROM WikiLog GROUP BY title HAVING sum(n) > 300));

ALTER TABLE public.wikilog OWNER TO akseewa11;

--
-- Name: public; Type: ACL; Schema: -; Owner: postgres
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM postgres;
GRANT ALL ON SCHEMA public TO postgres;
GRANT ALL ON SCHEMA public TO pgsql;
GRANT ALL ON SCHEMA public TO PUBLIC;


--
-- PostgreSQL database dump complete
--

