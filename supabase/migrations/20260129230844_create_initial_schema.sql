CREATE TABLE raw_news (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),                
    topic TEXT NOT NULL,               
    source VARCHAR(20),                
    title TEXT NOT NULL,
    url TEXT NOT NULL, 
    url_hash VARCHAR(64) NOT NULL,     
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT url_hash_unique UNIQUE (url_hash)
);

CREATE TABLE news_analysis (
    news_id INTEGER PRIMARY KEY REFERENCES raw_news(id) ON DELETE CASCADE,
    summary TEXT,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE high_water_mark (
    id SERIAL PRIMARY KEY,
    pipeline_name TEXT UNIQUE,
    last_processed_id INTEGER DEFAULT 0
);

INSERT INTO high_water_mark (pipeline_name, last_processed_id) 
VALUES ('news_analyser', 0) ON CONFLICT DO NOTHING;