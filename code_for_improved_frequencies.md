# Code to replicate the study




## Text preprocessing


The corpus used in this study can be downloaded from the GitHub repository: [https://github.com/computationalstylistics/100_english_novels](https://github.com/computationalstylistics/100_english_novels). Change the working directory so that the subfolder `corpus` can be seen by R:


``` R
setwd("100_english_novels")
```


In the preprocessing step, the package `stylo` will be used: it allows for loading the text files, tokenization, computing word frequencies, and preparing a document-term matrix. The package `tm` can be used instead but the code needs to be adjusted accordingly.

For inferring the word vectors, the package `text2vec` will be used.

``` R
library(text2vec)
library(stylo)
```

Now, loading a corpus from text files. The table of absolute frequencies (i.e. raw occurrences) will be build, as well as the table of relative frequencies.


``` R
# loading the corpus
texts = load.corpus.and.parse(files = "all", corpus.dir = "corpus")
# getting a genral frequency list
freq_list = make.frequency.list(texts, head = 1000)
# preparing the document-term matrix:
word_frequencies = make.table.of.frequencies(corpus = texts, features = freq_list, relative = FALSE)
word_frequencies_orig = make.table.of.frequencies(corpus = texts, features = freq_list, relative = TRUE)
```


Word vectors should follow now. [The code will be updated sooner, nonetheless the final result can be retrieved from the same GitHub repository containing 100 English novels (link posted above):]


``` R
load("100_english_novels/word_embedding_models/100_English_novels_GloVe_100_dimensions.RData")
```





## Computing word similarities

For each word, provide its semantic background, i.e. _n_ most similar words. This steps needs to be done once, for a given language. As with contemporary language models, the following matrix can be reused in many applications. Here, for the sake of the experiment, the whole range of >35,000 neighboring words has been computed, whereas in actual applications – as the current study clearly suggests – suffice it to have a background of 100 adjacent words. Here's the code for the whole stuff:


``` R
library(text2vec)

neighborship_size = 1000
#vocabulary = freq_list[1:1000]
vocabulary = rownames(word_vectors)


word_similarities_ALL = c()
#
for(word in vocabulary) {
    current_vector = word_vectors[word, , drop = FALSE]
    cos_sim = sim2(x = word_vectors, y = current_vector, method = "cosine", norm ="l2")
    similar_words = names(sort(cos_sim[,1], decreasing = TRUE)[1:neighborship_size])
    word_similarities_ALL = rbind(word_similarities_ALL, similar_words)
}

rownames(word_similarities_ALL) = vocabulary

# since the procedure puts the reference word as the most similar
# to itself, let's get rid of the first column
word_similarities_ALL = word_similarities_ALL[,-c(1)]
#
#class(word_similarities_ALL) = "stylo.data"
#save(word_similarities_ALL, file = "ranked_word_similarities.RData")
```


Please keep in mind that the procedure is rather costly. Conveniently, the results of the above function can be loaded from the current GitHub repository, namely from the following file:

``` R
load("ranked_word_similarities.RData")
```

It's a rather large file (>185Mb) and it will never be used in its entirety outside the current experiment. In real-life applications, a ranked list of 35,000 neighboring words could safely be trimmed to 100 or so neighbors. With this in mind a radically smaller file has been put into the repository; take this one for your actual stylometric computations:

``` R
load("ranked_word_similarities_100.RData")
```


## Core procedure: getting the improved frequencies

The function to compute relative word frequencies using a subset of reference words. Rather than dividing a given word's occurrences by the total number of words, it takes as a denominator the sum of the occurrences of _n_ relevant words. The relevant words need to be identified beforehand, e.g. using a word vector model.

In short, this is the core function of the whole procedure, and the only one that should be used routinely outside this experiment. So far, it's a prototype rather than an optimized fully-fledged function, yet even now it does the trick:


``` R
# word_frequencies - a table (matrix) with word occurrencies in a corpus
# word_vector_similarities - a table containig most similar words, in a descending order
# no_of_similar_words = the depth of density space, or the number of words to consider

compute_subset_frequencies = function(dtm_matrix, 
	                                  word_vector_similarities, 
	                                  no_of_similar_words) {

	semantic_space = word_vector_similarities[ , 1:no_of_similar_words, drop = FALSE]
	no_of_words = dim(semantic_space)[1]
	final_frequency_matrix = matrix(nrow = dim(dtm_matrix)[1], ncol = no_of_words)

	for(i in 1:no_of_words) {

		# check if the required word(s) appears in the corpus
	    words_sanitize = semantic_space[i,] %in% colnames(dtm_matrix)
	    words_to_compute = semantic_space[i, words_sanitize]
	    # if the corpus doesn't contain any of the words required 
	    # by the model, then grab the the most frequent word
	    # for reference (it should not happen often, though)
	    if(length(words_to_compute) == 0) {
	    	words_to_compute = colnames(dtm_matrix)[1]
	    }
	    # add the occurences of the current word being computed;
	    # e.g. for the word "of", add "of" to the equation
	    words_to_compute = c(colnames(dtm_matrix)[i], words_to_compute)
	    # getting the occurrences of the relevant words from
	    # the input matrix of word occurrences:
		f = dtm_matrix[, words_to_compute]
		# finally, computing new relative frequencies
		final_frequency_matrix[,i] = f[,1] / rowSums(f)

	}

	# sanitizing again, by replacing NaN values with Os
	final_frequency_matrix[is.nan(final_frequency_matrix)] = 0
	# tweaking the names of the rows and columns
	rownames(final_frequency_matrix) = rownames(dtm_matrix)
	colnames(final_frequency_matrix) = rownames(semantic_space)

	class(final_frequency_matrix) = "stylo.data"

return(final_frequency_matrix)
}
```



For the sake of the experiment, we'll also need a function to select training and test samples from the corpus. Here goes 
an auxiliary function to randomly pick training set texts in a stratified cross-validation scenario:

``` R
pick_training_texts = function(available_texts) {

	classes_all = gsub("_.*", "", available_texts)
	classes_unique = table(classes_all)
	classes_trainable = classes_unique[classes_unique > 1]
	texts_in_training_set = c()
	texts_in_test_set = c()

	for(current_class in names(classes_trainable)) {

	    texts_in_current_class = available_texts[classes_all == current_class]
	    texts_random_order = sample(texts_in_current_class)

	    #add_to_test_set = texts_random_order[1]
	    #add_to_training_set = texts_random_order[2:length(texts_random_order)]
	    add_to_training_set = texts_random_order[1]

	    texts_in_training_set = c(texts_in_training_set, add_to_training_set)

	}
	return(texts_in_training_set)
}
```






## Supervised classification


The main classification procedure follows. It iterates over the increasing number of nearest neighbors for each analyzed word. Specifically, for each of the MFWs, it fist looks for its 1 nearest neighbor, then 2 nearest neighbors, all the way to 10,000. For each iteration, a supervised text classification takes place.

The function requires:
* `word_frequencies` -- a document-term matrix with raw frequencies (occurrences) of words across a given corpus. Such a matrix can be produced either by the package `stylo` (as above), or by the package `tm` (not covered here).
* `word_similarities_ALL` -- a table with nearest neighbors, e.g. the row for the word _the_ contains: "the", "of", "this", "in", "there", "on", ...
* the function `compute_subset_frequencies` as defined above, to perform relative frequency calculations on a subset of reference words
* the function `pick_training_texts` as defined above, to randomly select training set members in a stratified cross-validation scenario.




``` R
mfw_coverage = seq(100, 1000, 50)
method = "delta"
distance = "wurzburg"

semantic_area_to_cover = c(1:9, seq(10, 90, 10), seq(100, 900, 100), seq(1000, 10000, 1000))
collect_results_all_similarity_areas = c()


for(surrounding_words in semantic_area_to_cover) {

	p = compute_subset_frequencies(word_frequencies, word_similarities_ALL, surrounding_words)
	# now the main procedure takes place:
    texts_in_training_set = pick_training_texts(rownames(p))
    pinpointed_training_texts = rownames(p) %in% texts_in_training_set
    training_set_p = p[pinpointed_training_texts, ]
    test_set_p = p[!pinpointed_training_texts, ]

    collect_results = c()
    for(mfw in mfw_coverage) {

    	results = classify(gui = FALSE, training.frequencies = training_set_p, test.frequencies = test_set_p, mfw.min = mfw, mfw.max = mfw, cv.folds = 100, classification.method = method, distance.measure = distance)
    	f1 = performance.measures(results)$avg.f
        collect_results = c(collect_results, f1)
    }
    collect_results_all_similarity_areas = cbind(collect_results_all_similarity_areas, collect_results)

}


rownames(collect_results_all_similarity_areas) = mfw_coverage
colnames(collect_results_all_similarity_areas) = semantic_area_to_cover

#save(collect_results_all_similarity_areas, file = "performance_wurzburg.RData")

```


## Baseline


Additionally, it makes a lot of sense to compute a baseline, i.e. a series of tests based on standard relative frequencies. On theoretical grounds, one round of tests for 100, 200, ..., 1000 MFWs should be sufficient, but it will be more reliable if the baseline computation is repeated independently _n_ times (here, 37 times, in order to match the size of the above `semantic_area_to_cover` vector). Please keep in mind that all the parameters are inherited from the above setup, in order to match the baseline with the actual results.



``` R
mfw_coverage = seq(100, 1000, 50)
method = "delta"
distance = "wurzburg"

collect_baseline_results = c()

for(dummy_iterations in 1:37) {

	results_baseline_ALL = c()

	# splitting the data into the training set and the test set
	texts_in_training_set = pick_training_texts(rownames(word_frequencies_orig))
	pinpointed_training_texts = rownames(word_frequencies_orig) %in% texts_in_training_set
	training_set = word_frequencies_orig[pinpointed_training_texts, ]
	test_set = word_frequencies_orig[!pinpointed_training_texts, ]

	for(mfw in mfw_coverage) {

		# now adding a traditional MFW approach as a baseline
		results_baseline = classify(gui = FALSE, training.frequencies = training_set, test.frequencies = test_set, mfw.min = mfw, mfw.max = mfw, cv.folds = 100, classification.method = method, distance.measure = distance)
		f1_baseline = performance.measures(results_baseline)$avg.f
		results_baseline_ALL = c(results_baseline_ALL, f1_baseline)

	}

collect_baseline_results = cbind(collect_baseline_results, results_baseline_ALL)

}

rownames(collect_baseline_results) = mfw_coverage
colnames(collect_baseline_results) = 1:37

#save(collect_baseline_results, file = "results_baseline_wurzburg.RData")

```



## Plotting the results



``` R
load("results_wurzburg.RData")
load("results_baseline_wurzburg.RData")
# define the gain of performance as a difference between the final results and the baseline (both expressed in f1 scores)
performance_gain = collect_results_all_similarity_areas - collect_baseline_results
# get rid of the values that are worse than the baseline
performance_gain[performance_gain <= 0] = NA
```




``` R
library(lattice)
library(viridisLite)
# setting the scales
x.scale <- list(at = c(1, 5, 10, 14, 19, 23, 28, 32, 36), label = c(1, 5, 10, 50, 100, 500, 1000, 5000, 10000))
y.scale <- list(at = c(1, 4, 7, 10, 13, 16, 19), labels = seq(100, 1000, 150))
# absolute values
plot_1 = levelplot(t(collect_results_all_similarity_areas), cuts = 1000, scales = list(x = x.scale, y = y.scale), col.regions = rev(turbo(1000)), xlab= "semantic background (words)", ylab = "number of MFWs")
# gain over the baseline
plot_2 = levelplot(t(performance_gain), cuts = 1000, scales = list(x = x.scale, y = y.scale), col.regions = rev(turbo(1000, end = 0.6)), xlab= "semantic background (words)", ylab = "MFWs")

```

``` R
png(file = "performance_wurzburg.png", res = 300, width = 6, height = 4, units = "in")
plot_1
dev.off()

png(file = "performance_gain_wurzburg.png", res = 300, width = 6, height = 4, units = "in")
plot_2
dev.off()
```


Modified colors:

``` R
# setting the scales
x.scale <- list(at = c(1, 5, 10, 14, 19, 23, 28, 32, 36), label = c(1, 5, 10, 50, 100, 500, 1000, 5000, 10000))
y.scale <- list(at = c(1, 4, 7, 10, 13, 16, 19), labels = seq(100, 1000, 150))
# absolute values
plot_1 = levelplot(t(collect_results_all_similarity_areas), cuts = 1000, scales = list(x = x.scale, y = y.scale), col.regions = colorRampPalette(c("blue", "yellow", "red", "black")), xlab= "semantic background (words)", ylab = "number of MFWs")
# gain over the baseline
plot_2 = levelplot(t(performance_gain), cuts = 1000, scales = list(x = x.scale, y = y.scale), col.regions = colorRampPalette(c("yellow", "red", "black")), xlab= "semantic background (words)", ylab = "MFWs")

```







## Experiment 2: semantic neighbors defined by distances

Rather than selecting _n_ words, this approach is aimed at testing similarity of words at a given _distance_ from the seed word. As will become clear later on, this is perhaps not the best way to extract the semantic neighborhood, yet it makes a lot of sense examine it anyway.

First, we need information -- for each word -- what is the number of the word's neighbors at the cosine distance of 0.9, 0.85, 0.7, ..., all the way to -1 (since the cosine distances for this particular model cover the range {-1,1} rather than {0,1}). Resulting is a matrix: words in columns, cosine distance in rows, the number of respective neighboring words in particular cells.




``` R
#
load("word_embedding_models/100_English_novels_GloVe_100_dimensions.Rdata")

#vocabulary = c("the", "i", "and", "of", "to", "a", "it", "had", "was", "as", "my", "his")

vocabulary = freq_list[1:1000]
similarity_thresholds_to_assess = rev(seq(-1, 0.9, 0.05))
similarity_neighborhood_ALL = c()

message("")
count = 0


for(word in vocabulary) {

    count = count +1
    if(count %% 10 == 0) message(".", appendLF = FALSE)

    current_vector = word_vectors[word, , drop = FALSE]
    cos_sim = sim2(x = word_vectors, y = current_vector, method = "cosine", norm ="l2")
    similarity_neighborhood = c()

    for(similarity_threshold in similarity_thresholds_to_assess) {
        no_of_similar_words = sum(as.numeric(cos_sim[,1] >= similarity_threshold))#length(cos_sim[cos_sim[,1] >= similarity_threshold])
        similarity_neighborhood = c(similarity_neighborhood, no_of_similar_words)
    }

    similarity_neighborhood_ALL = rbind(similarity_neighborhood_ALL, similarity_neighborhood)
}

similarity_neighborhood_ALL = t(similarity_neighborhood_ALL)
colnames(similarity_neighborhood_ALL) = vocabulary
rownames(similarity_neighborhood_ALL) = round(similarity_thresholds_to_assess, 2)

# save(similarity_neighborhood_ALL, file = "areas_of_cosine_similarity.RData")

```


Given the information of how many neighboring words should be taken into account (e.g. the word "the" has 38 neighbors within the distance of 0.6), and given the previously computed information of the order of neighbors for each seed word (e.g. for the word "the" these are: "of", "this", "in", "there", ...), we can get the _n_ neighbors for a given distance. The following function is responsible for it:


``` R
# word_frequencies - a table (matrix) with word occurrencies in a corpus
# word_vector_similarities - a table containig most similar words, in a descending order
# no_of_similar_words = the depth of density space, or the number of words to consider

compute_subset_frequencies_B = function(word_frequencies, 
	                                  word_vector_similarities, 
	                                  no_of_similar_words) {

	if(max(no_of_similar_words) == dim(word_vector_similarities)[2]) {
		semantic_space = word_vector_similarities[,1:max(no_of_similar_words)]
	} else {
		semantic_space = word_vector_similarities[,1:(max(no_of_similar_words)+1)]
	}


	no_of_words = dim(semantic_space)[1]
	final_frequency_matrix = matrix(nrow = dim(word_frequencies)[1], ncol = no_of_words)

	for(i in 1:no_of_words) {

	    words_sanitize = semantic_space[i,] %in% colnames(word_frequencies)
	    if(max(no_of_similar_words) < dim(word_vector_similarities)[2]) {
	        words_sanitize[(no_of_similar_words[i] +1):(length(words_sanitize))] = FALSE
	    }
	    words_to_compute = semantic_space[i, words_sanitize]
	    if(length(words_to_compute) == 1) {
	    	words_to_compute = c(words_to_compute, colnames(word_frequencies)[1])
	    }
		f = word_frequencies[, words_to_compute]
		final_frequency_matrix[,i] = f[,1] / rowSums(f)

	}

	# sanitizing again, by replacing NaN values with Os
	final_frequency_matrix[is.nan(final_frequency_matrix)] = 0
	# tweaking the names of the rows and columns
	rownames(final_frequency_matrix) = rownames(word_frequencies)
	colnames(final_frequency_matrix) = rownames(semantic_space)

return(final_frequency_matrix)
}

```


Finally, we can assess the classification performance (f1 scores in this case) for different vectors of MFWs's frequencies relative to different semantic areas:


``` R
########### the main iterative procedure

mfw_coverage = seq(100,1000,50)
method = "delta"
distance = "wurzburg"

cosine_similarity_to_cover = round(rev(seq(-1, 0.9, 0.05)), 2)
collect_results_all_similarity_areas = c()


for(surrounding_word_area in cosine_similarity_to_cover) {

	p = compute_subset_frequencies_B(word_frequencies, word_similarities_ALL, similarity_neighborhood_ALL[as.character(surrounding_word_area),])
	# now the main procedure takes place:
    texts_in_training_set = pick_training_texts(rownames(p))
    pinpointed_training_texts = rownames(p) %in% texts_in_training_set
    training_set_p = p[pinpointed_training_texts, ]
    test_set_p = p[!pinpointed_training_texts, ]

    collect_results = c()
    for(mfw in mfw_coverage) {

    	results = classify(gui = FALSE, training.frequencies = training_set_p, test.frequencies = test_set_p, mfw.min = mfw, mfw.max = mfw, cv.folds = 100, classification.method = method, distance.measure = distance)
    	f1 = performance.measures(results)$avg.f
        collect_results = c(collect_results, f1)
    }
    collect_results_all_similarity_areas = cbind(collect_results_all_similarity_areas, collect_results)

}

rownames(collect_results_all_similarity_areas) = mfw_coverage
colnames(collect_results_all_similarity_areas) = cosine_similarity_to_cover

#save(collect_results_all_similarity_areas, file = "results_areas_of_cosine_similarity_wurzburg.RData")
```


Plotting the results: basically the same code as above is applied, except that the scales have to be slightly adjusted:



``` R
load("results_areas_of_cosine_similarity_wurzburg.RData")
load("results_baseline_wurzburg.RData")
# in order to be compliant with the previous plots, let's trim the last two columns of the results' matrix (the results for -0.95 and -1 are meaningless anyway...):
collect_results_all_similarity_areas = collect_results_all_similarity_areas[,1:37]
# define the gain of performance as a difference between the final results and the baseline (both expressed in f1 scores)
performance_gain = collect_results_all_similarity_areas - collect_baseline_results
# get rid of the values that are worse than the baseline
performance_gain[performance_gain <= 0] = NA
```



``` R
library(lattice)
library(viridisLite)
# setting the scales
x.scale <- list(at = seq(1, 37, 4), label = seq(0.9, -1, -0.2))
y.scale <- list(at = c(1, 4, 7, 10, 13, 16, 19), labels = seq(100, 1000, 150))
# absolute values
plot_1 = levelplot(t(collect_results_all_similarity_areas), cuts = 1000, scales = list(x = x.scale, y = y.scale), col.regions = rev(turbo(1000)), xlab= "semantic background (cosine distance)", ylab = "number of MFWs")
# gain over the baseline
plot_2 = levelplot(t(performance_gain), cuts = 1000, scales = list(x = x.scale, y = y.scale), col.regions = rev(turbo(1000, end = 0.6)), xlab= "semantic background (cosine distance)", ylab = "MFWs")

```


``` R
png(file = "overall_wv_distance_wurzburg.png", res = 300, width = 6, height = 4, units = "in")
plot_1
dev.off()

png(file = "performance_gain_wv_distance_wurzburg.png", res = 300, width = 6, height = 4, units = "in")
plot_2
dev.off()
```



