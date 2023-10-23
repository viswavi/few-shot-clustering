def select_keyphrase_expansion_prompt(dataset_name):
    if dataset_name == "OPIEC59k":
        return """I am trying to cluster entity strings on Wikipedia according to the Wikipedia article title they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. Entities may be weirdly truncated or ambiguous - e.g. "Wind" may refer to the band "Earth, Wind, and Fire" or to "rescue service". For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Generate a comprehensive set of alternate entity names as a JSON-formatted list.

Entity: "fictional character"

Context Sentences:
1) "Camille Raquin is a fictional character created by Émile Zola ."
2) "Druu is a fictional character appearing in comic books published by DC Comics ."
3) "Mallen is a fictional character that appears in comic books published by Marvel Comics .""

Keyphrases: ["fictional characters", "characters", "character"]

Entity: "Catholicism"

Context Sentences:
1) "Years after Anne had herself converted , James avowed his Catholicism , which was a contributing factor to the Glorious Revolution ."
2) "The `` Catechism of the Catholic Church '' , representing Catholicism 's great regard for Thomism , the teachings of St. Thomas Aquinas , affirms that it is a Catholic doctrine that God 's existence can indeed be demonstrated by reason ."
3) "Palestinian Christians belong to one of a number of Christian denominations , including Eastern Orthodoxy , Oriental Orthodoxy , Catholicism ( Eastern and Western rites ) , Anglicanism , Lutheranism , other branches of Protestantism and others .""

Keyphrases: ["Catholic Church", "Roman Catholic", "Catholic"]

Entity: "Wind"

Context Sentences:
1) "It was co-produced by Earth , Wind & Fire 's keyboardist Larry Dunn ."
2) "Guitarist Tom Morello described the sound as `` Earth , Wind and Fire meets Led Zeppelin '' ."
3) "Taylor Mesplé also sang with Lampa along with Sydney Hostetler and the late Winston Ford ( Earth , Wind & Fire , The Drifters ) .""

Keyphrases: ["Earth & Fire", "Earth", "Wind & Fire"]

Entity: "Elizabeth"

Context Sentences:
1) "He also performed at the London Palladium for Queen Elizabeth ."
2) "On July 5 , 2010 , Ray was honored to be a host of the informal lunch for Queen Elizabeth 's visit to Toronto ."
3) "Its 1977 premiere was staged at the Royal Festival Hall in London as part of Queen Elizabeth 's Silver Jubilee .""

Keyphrases: ["Elizabeth II", "HM"]

Entity: "Elizabeth II"""
    elif dataset_name == "reverb45k":
        return """I am trying to cluster entity strings from the Internet according to the Freebase knowledge graph entity that they refer to. To help me with this, for a given entity name, please provide me with a comprehensive set of alternative names that could refer to the same entity. For each entity, I will provide you with a sentence where this entity is used to help you understand what this entity refers to. Some entities have no alternate names, while others will have alternate names not directly mentioned in the context sentences. Generate a comprehensive set of alternate entity names as a JSON-formatted list.

Entity: "Hank Aaron"

Context Sentences:
1) "Hank Aaron was born in Mobile"
2) "Hank Aaron (1934-) was born in Mobile, AL on February 5, 1934."
3) "**** Mobile, Alabama Career highlights and awards 21x All-Star selection (1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975) 1957 NL MVP 1970 Lou Gehrig Memorial Award World Series champion (1957) 3rd on all-time hits list with 3,771 MLB Records: Member of the National Baseball Hall of Fame Early life Hank Aaron was born in Mobile, Alabama to Herbert and Estella Aaron.""

Keyphrases: ["Henry Aaron", "Aaron"]

Entity: "Apple"

Context Sentences:
1) "Apple buy Sony"
2) "Apple uses AAC"
3) "Apple does on OSX""

Keyphrases: ["Apple Corps", "Apple Computer", "Apple Inc.", "Apple II"]

Entity: "Jason"

Context Sentences:
1) "Jason is taking on Google"
2) "Jason turned to face Bruce"
3) "Jason has ties to Brad Pitt""

Keyphrases: []

Entity: "Insomniac Games"

Context Sentences:
1) "Insomniac Games operates out of Burbank"
2) "Insomniac Games operates out of Burbank, California."
3) "Insomniac Games operates out of Burbank, California. In June 2008, the company announced an expansion studio in the Raleigh-Durham area of North Carolina (commencing January 2009.)""

Keyphrases: ["Insomniac"]"""
    elif dataset_name == "tweet":
        return """I am trying to cluster tweets based on whether they discuss the same topic. To do this, given a (stopword-removed) tweet, please provide a comprehensive set of Keyphrases or keyphrases that could describe this tweet's topic. These keyphrases should be distinct from those that might describe tweets with different topics. Since the tweets already look like keyphrases, feel free to include keyphrases not listed in the tweet, and don't feel like you need to include very many of the original words from the tweet. Generate a comprehensive set of keyphrases as a JSON-formatted list.

Tweet: "brain fluid buildup delay giffords rehab"

Keyphrases: ["gabrielle giffords", "giffords recovery"]

Tweet: "trailer talk week movie rite mechanic week opportunity"

Keyphrases: ["movies", "in theaters", "trailer talk"]

Tweet: "gbagbo camp futile cut ivory coast economy"

Keyphrases: ["gbagbo", "ivory coast"]

Tweet: "chicken cavatelli soup"

Keyphrases: ["cooking", "tasty recipes"]"""
    elif dataset_name == "clinc":
        return """I am trying to cluster task-oriented dialog system queries based on whether they express the same general user intent. To help me with this, for a given user query, provide a comprehensive set of keyphrases that could describe this query's intent. These keyphrases should be distinct from those that might describe queries with different intents. Generate the set of keyphrases as a JSON-formatted list.

Query: "how would you say fly in italian"

Keyphrases: ["translation", "translate"]

Query: "what does assiduous mean"

Keyphrases: ["definition", "define"]

Query: "find my cellphone for me!"

Keyphrases: ["location", "find", "locate", "tracking", "track"]"""
    elif dataset_name == "bank77":
        return """I am trying to cluster queries for a online banking system based on whether they express the same general user intent. To help me with this, for a given banking query, provide a comprehensive set of keyphrases that could describe this query's intent. These keyphrases should be distinct from those that might describe banking-related queries with different intents. Generate the set of keyphrases as a JSON-formatted list.

Query: "How do I locate my card?"

Keyphrases: ["card status", "status update", "card location"]

Query: "Whats the delivery time to the United States?"

Keyphrases: ["delivery time", "ETA", "card delivery"]

Query: "Can you cancel my purchase?"

Keyphrases: ["cancel purchase", "refund"]

Query: "Why don't I have my transfer?"

Keyphrases: ["transfer", "transfer failed"]"""
    else:
        raise ValueError(f"Invalid dataset given: {dataset_name} not found")