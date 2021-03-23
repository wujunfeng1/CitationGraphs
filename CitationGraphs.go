// CitationGraphs.go project CitationGraphs.go
package CitationGraphs

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/wujunfeng1/ConcurrenceBasedClustering"
	"github.com/wujunfeng1/KeyphraseExtraction"
)

var reUnicodeHex *regexp.Regexp
var reUnicodeDec *regexp.Regexp

func init() {
	reUnicodeHex = regexp.MustCompile("&//[Xx]([A-Fa-f0-9])+;")
	reUnicodeDec = regexp.MustCompile("&//([0-9])+;")
	rand.Seed(time.Now().Unix())
}

// =================================================================================================
// struct CitationNode
// brief description: The node structure of a citation graph, the data of which are collected
//                    from crawling the website https://academic.microsoft.com , which is the
//                    public service website of MAG(Microsoft Academic Graph)
// fields:
//   id: The MAG(Microsoft Academic Graph) id of the paper at this node.
//       We can access the detail of the paper with the id by navigating the web link:
//       https://academic.microsoft.com/paper/$id
//   year: The year of publication of this paper.
//   title: The title of the paper.
//   labels: The labels from MAG(Microsoft Academic Graph).
//   refs: The references of this paper collected from MAG.
//         Please note that this field could be inaccurate: MAG suffers delay of info.
//         Many new papers with refs don't have refs listed in MAG.
//   cites: The citations to this paper from other papers (also collected from MAG).
//         Please note that this field is as inaccurate as refs due to the same reason.
type CitationNode struct {
	ID     int64
	Year   int64
	Title  string
	Labels []string
	Refs   []int64
	Cites  []int64
}

// =================================================================================================
// struct CitationGraph
// brief description: A data structure of citation graph, the data of which are collected from
//                    the website https://academic.microsoft.com , which is the public service
//                    website of MAG(Microsoft Academic Graph)
// fields:
//   nodes: The node of struct CitationNode stored in a dictionary with key = id
//   toBeAnalyzed: The list of nodes to be analyzed. In ijcai_clustering dataset, these nodes
//                 are those published in IJCAI.
type CitationGraph struct {
	Nodes        map[int64]*CitationNode
	ToBeAnalyzed []int64
	idxMainNodes map[int64]int
}

// =================================================================================================
// struct Corpus
// brief description: the corpus data structure
type Corpus struct {
	Vocab map[string]int // the vocabulary
	Docs  []map[int]int  // keys: docID, wordID, value: word count
}

// =================================================================================================
// struct CorpusX
// brief description: the extended corpus data structure
type CorpusX struct {
	Vocab map[string]int  // the vocabulary
	Docs  [][]map[int]int // keys: docID, wordID, value: word count
}

// =================================================================================================
// func NewCorpus
// brief description: create an empty corpus
func NewCorpus() *Corpus {
	return &Corpus{
		Vocab: map[string]int{},
		Docs:  []map[int]int{},
	}
}

// =================================================================================================
// func NewCorpusX
// brief description: create an empty corpus
func NewCorpusX() *CorpusX {
	return &CorpusX{
		Vocab: map[string]int{},
		Docs:  [][]map[int]int{},
	}
}

// =================================================================================================
// func (this *Corpus) AddDoc
// brief description: add one document to corpus with specified docId and word count list, if the
// 	specified docId already exists in corpus, the old doc will be overwritted
func (this *Corpus) AddDoc(words []string) {
	// ---------------------------------------------------------------------------------------------
	// step 1: create a new doc and count word counts
	doc := map[int]int{}
	for _, word := range words {
		wordID, exists := this.Vocab[word]
		if !exists {
			wordID = len(this.Vocab)
			this.Vocab[word] = wordID
		}
		count, exists := doc[wordID]
		if !exists {
			count = 0
		}
		doc[wordID] = count + 1
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: add the new doc into docs
	this.Docs = append(this.Docs, doc)
}

// =================================================================================================
// func (this *CorpusX) AddDoc
// brief description: add one document to corpus with specified docId and word count list, if the
// 	specified docId already exists in corpus, the old doc will be overwritted
func (this *CorpusX) AddDoc(words [][]string) {
	// ---------------------------------------------------------------------------------------------
	// step 1: create a new doc and count word counts
	doc := make([]map[int]int, len(words))
	for idx, wordGroup := range words {
		doc[idx] = map[int]int{}
		for _, word := range wordGroup {
			wordID, exists := this.Vocab[word]
			if !exists {
				wordID = len(this.Vocab)
				this.Vocab[word] = wordID
			}
			count, exists := doc[idx][wordID]
			if !exists {
				count = 0
			}
			doc[idx][wordID] = count + 1
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: add the new doc into docs
	this.Docs = append(this.Docs, doc)
}

// =================================================================================================
// func (this *Corpus) GetConcurrences
// brief description: get concurrences from corpus
func (this *Corpus) GetConcurrences() map[uint]map[uint]float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: create concurrences
	concurrences := map[uint]map[uint]float64{}

	// ---------------------------------------------------------------------------------------------
	// step 2: spawn goroutines to count concurrences when words are in the same document
	numCPUs := runtime.NumCPU()
	numDocs := len(this.Docs)
	chI := make(chan int)
	chWorkers := make(chan bool)
	lock := sync.RWMutex{}
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			myConcurrences := map[uint]map[uint]float64{}
			for i0 := range chI {
				i1 := i0 + 100
				if i1 > numDocs {
					i1 = numDocs
				}
				for i := i0; i < i1; i++ {
					wordCount := this.Docs[i]
					for w1, count1 := range wordCount {
						concurrencesOfW1, exists := myConcurrences[uint(w1)]
						if !exists {
							concurrencesOfW1 = map[uint]float64{}
							myConcurrences[uint(w1)] = concurrencesOfW1
						}
						for w2, count2 := range wordCount {
							if w1 != w2 {
								count, exists := concurrencesOfW1[uint(w2)]
								if !exists {
									count = 0.0
								}
								concurrencesOfW1[uint(w2)] = count + float64(count1*count2)
							}
						}
					}
				}
			}

			// now we merge the results into concurrences
			lock.Lock()
			defer lock.Unlock()
			for key1, value1 := range myConcurrences {
				row, exists := concurrences[key1]
				if !exists {
					concurrences[key1] = value1
				} else {
					for key2, value2 := range value1 {
						oldValue, exists := row[key2]
						if !exists {
							oldValue = 0.0
						}
						row[key2] = oldValue + value2
					}
				}
			}

			// tell main thread that this worker finishes
			chWorkers <- true
		}()
	}

	// ---------------------------------------------------------------------------------------------
	// step	3: send tasks through chI
	for i := 0; i < numDocs; i += 100 {
		chI <- i
	}
	close(chI)

	// ---------------------------------------------------------------------------------------------
	// step 4: wait for all workers
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}

	// ---------------------------------------------------------------------------------------------
	// step 5: return the concurrences
	return concurrences
}

// =================================================================================================
// func (this *CorpusX) GetExclusions
// brief description: get exclusions from corpus
func (this *CorpusX) GetExclusions() map[uint]map[uint]bool {
	// ---------------------------------------------------------------------------------------------
	// step 1: create empty exclusions
	exclusions := map[uint]map[uint]bool{}

	// ---------------------------------------------------------------------------------------------
	// step 2: create texts
	numTexts := len(this.Vocab)
	texts := make([]string, numTexts)
	for text, id := range this.Vocab {
		texts[id] = text
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: spawn goroutines to find exclusions when words are in the same group
	numCPUs := runtime.NumCPU()
	numDocs := len(this.Docs)
	chI := make(chan int)
	chExclusions := make(chan map[uint]map[uint]bool)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			myExclusions := map[uint]map[uint]bool{}
			for i0 := range chI {
				i1 := i0 + 100
				if i1 > numDocs {
					i1 = numDocs
				}
				for i := i0; i < i1; i++ {
					wordGroups := this.Docs[i]
					for _, wordCount := range wordGroups {
						for w1, _ := range wordCount {
							exclusionsOfW1, exists := myExclusions[uint(w1)]
							if !exists {
								exclusionsOfW1 = map[uint]bool{}
								myExclusions[uint(w1)] = exclusionsOfW1
							}
							text1 := texts[w1]
							for w2, _ := range wordCount {
								if w2 != w1 {
									text2 := texts[w2]
									if KeyphraseExtraction.Overlaps(text1, text2) {
										exclusionsOfW1[uint(w2)] = true
									}
								}
							}
						}
					}

				}
			}

			// give main thread my results
			chExclusions <- myExclusions
		}()
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: send tasks through chI
	for i := 0; i < numDocs; i += 100 {
		chI <- i
	}
	close(chI)

	// ---------------------------------------------------------------------------------------------
	// step 5: merge results
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		myExclusions := <-chExclusions
		for w1, deltaExclusionsOfW1 := range myExclusions {
			exclusionsOfW1, exists := exclusions[w1]
			if exists {
				for w2, _ := range deltaExclusionsOfW1 {
					exclusionsOfW1[w2] = true
				}
			} else {
				exclusions[w1] = deltaExclusionsOfW1
			}
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 6: return results
	return exclusions
}

// =================================================================================================
// func (this *CorpusX) GetConcurrences
// brief description: get concurrences from corpus
func (this *CorpusX) GetConcurrences() map[uint]map[uint]float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: create empty concurrences
	concurrences := map[uint]map[uint]float64{}

	// ---------------------------------------------------------------------------------------------
	// step 2: spawn goroutines to count concurrences when words are in the same document
	numCPUs := runtime.NumCPU()
	numDocs := len(this.Docs)
	chI := make(chan int)
	chWorkers := make(chan bool)
	lock := sync.RWMutex{}
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			myConcurrences := map[uint]map[uint]float64{}
			for i0 := range chI {
				i1 := i0 + 100
				if i1 > numDocs {
					i1 = numDocs
				}
				for i := i0; i < i1; i++ {
					wordGroups := this.Docs[i]
					for idx1, wordCount1 := range wordGroups {
						for w1, count1 := range wordCount1 {
							concurrencesOfW1, exists := myConcurrences[uint(w1)]
							if !exists {
								concurrencesOfW1 = map[uint]float64{}
								myConcurrences[uint(w1)] = concurrencesOfW1
							}
							for idx2, wordCount2 := range wordGroups {
								// skip same word group
								if idx1 == idx2 {
									continue
								}

								for w2, count2 := range wordCount2 {
									if w1 != w2 {
										count, exists := concurrencesOfW1[uint(w2)]
										if !exists {
											count = 0.0
										}
										concurrencesOfW1[uint(w2)] = count + float64(count1*count2)
									}
								}
							}
						}
					}

				}
			}

			// now we merge the results into concurrences
			lock.Lock()
			defer lock.Unlock()
			for key1, value1 := range myConcurrences {
				row, exists := concurrences[key1]
				if !exists {
					concurrences[key1] = value1
				} else {
					for key2, value2 := range value1 {
						oldValue, exists := row[key2]
						if !exists {
							oldValue = 0.0
						}
						row[key2] = oldValue + value2
					}
				}
			}

			// tell main thread that this worker finishes
			chWorkers <- true
		}()
	}

	// ---------------------------------------------------------------------------------------------
	// step	3: send tasks through chI
	for i := 0; i < numDocs; i += 100 {
		chI <- i
	}
	close(chI)

	// ---------------------------------------------------------------------------------------------
	// step 4: wait for all workers
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}

	// ---------------------------------------------------------------------------------------------
	// step 5: return the concurrences
	return concurrences
}

// =================================================================================================
// func (this *Corpus) translate
func (this *Corpus) translate(subCorpus *Corpus) *Corpus {
	oldSubVocab := subCorpus.Vocab
	translation := map[int]int{}
	for word, oldID := range oldSubVocab {
		newID, exists := this.Vocab[word]
		if !exists {
			log.Fatal("not a sub corpus")
		}
		translation[oldID] = newID
	}

	newDocs := make([]map[int]int, len(subCorpus.Docs))
	for docID, oldDoc := range subCorpus.Docs {
		newDoc := map[int]int{}
		for oldID, count := range oldDoc {
			newDoc[translation[oldID]] = count
		}
		newDocs[docID] = newDoc
	}

	return &Corpus{Vocab: this.Vocab, Docs: newDocs}
}

// =================================================================================================
// func (this *CorpusX) translate
func (this *CorpusX) translate(subCorpus *CorpusX) *CorpusX {
	oldSubVocab := subCorpus.Vocab
	translation := map[int]int{}
	for word, oldID := range oldSubVocab {
		newID, exists := this.Vocab[word]
		if !exists {
			log.Fatal("not a sub corpus")
		}
		translation[oldID] = newID
	}

	newDocs := make([][]map[int]int, len(subCorpus.Docs))
	for docID, oldDoc := range subCorpus.Docs {
		newDoc := make([]map[int]int, len(oldDoc))
		for idxGroup, oldWordGroup := range oldDoc {
			newDoc[idxGroup] = map[int]int{}
			for oldID, count := range oldWordGroup {
				newDoc[idxGroup][translation[oldID]] = count
			}
		}
		newDocs[docID] = newDoc
	}

	return &CorpusX{Vocab: this.Vocab, Docs: newDocs}
}

// =================================================================================================
// interface LDAModel
// brief description: the common interface of LDA models
type TopicModel interface {
	// train model for iter iteration
	Train(numIters int)
	// do inference for new doc with its wordCounts
	Infer(doc map[int]int) []float64
	// compute entropy
	ComputeEntropy() float64
	// compute relative entropy
	ComputeRelativeEntropy() float64
}

// =================================================================================================
// struct DocWord
type DocWord struct {
	DocId  int
	WordId int
}

// =================================================================================================
// struct LDA
// brief description: the data structure of LDA model with Collapsed Gibbs Sampler
// note:
//	The fast collapsed gibbs sampler algorithm can be found in reference:
//	Porteous, I., Newman, D., Ihler, A., Asuncion, A., Smyth, P., & Welling, M. (2008, August). Fast
//	collapsed gibbs sampling for latent dirichlet allocation. In Proceedings of the 14th ACM SIGKDD
//	international conference on Knowledge discovery and data mining (pp. 569-577).
type LDA struct {
	Alpha     float64 // document topic mixture hyperparameter
	Beta      float64 // topic word mixture hyperparameter
	NumTopics int     // number of topics

	Data *Corpus // the input corpus

	WordTopicCount [][]int           // word-topic count table
	DocTopicCount  [][]int           // doc-topic count table
	TopicCountSum  []int             // word-topic-sum count table
	DocWordToTopic map[DocWord][]int // doc-word-topic count table
}

// =================================================================================================
// func NewLDA
// brief description: create an LDA instance with collapsed gibbs sampler
func NewLDA(numTopics int, alpha float64, beta float64, data *Corpus) *LDA {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	if data == nil {
		log.Fatalln("corpus is nil")
	}

	if numTopics <= 0 {
		log.Fatalln("numTopics cannot <= 0.")
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: create wordTopicCount
	numWords := len(data.Vocab)
	wordTopicCount := make([][]int, numWords)
	for w := 0; w < numWords; w++ {
		topicCountOfW := make([]int, numTopics)
		wordTopicCount[w] = topicCountOfW
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: create docTopicCount
	numDocs := len(data.Docs)
	docTopicCount := make([][]int, numDocs)
	for doc := 0; doc < numDocs; doc++ {
		topicCountOfDoc := make([]int, numTopics)
		docTopicCount[doc] = topicCountOfDoc
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: create topicCountSum
	topicCountSum := make([]int, numTopics)

	// ---------------------------------------------------------------------------------------------
	// step 5: create docWordToTopic
	docWordToTopic := map[DocWord][]int{}
	for doc, wordCounts := range data.Docs {
		for w, count := range wordCounts {
			docWord := DocWord{doc, w}
			toTopic := make([]int, count)
			docWordToTopic[docWord] = toTopic
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 6: assemble the result
	result := &LDA{
		Alpha:     alpha,
		Beta:      beta,
		NumTopics: numTopics,
		Data:      data,

		WordTopicCount: wordTopicCount,
		DocTopicCount:  docTopicCount,
		TopicCountSum:  topicCountSum,
		DocWordToTopic: docWordToTopic,
	}

	// ---------------------------------------------------------------------------------------------
	// step 7: return the result
	return result
}

// =================================================================================================
// func (this *LDA) updateCounters
func (this *LDA) updateCounters() {
	// ---------------------------------------------------------------------------------------------
	// step 1: initialize Counters
	numTopics := this.NumTopics
	for idxTopic := 0; idxTopic < numTopics; idxTopic++ {
		this.TopicCountSum[idxTopic] = 0
	}

	numWords := len(this.Data.Vocab)
	for w := 0; w < numWords; w++ {
		topicCountOfW := this.WordTopicCount[w]
		for idxTopic := 0; idxTopic < numTopics; idxTopic++ {
			topicCountOfW[idxTopic] = 0
		}
	}

	numDocs := len(this.Data.Docs)
	for doc := 0; doc < numDocs; doc++ {
		topicCountOfDoc := this.DocTopicCount[doc]
		for idxTopic := 0; idxTopic < numTopics; idxTopic++ {
			topicCountOfDoc[idxTopic] = 0
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: count them!
	for doc, wordCounts := range this.Data.Docs {
		for w, count := range wordCounts {
			toTopic := this.DocWordToTopic[DocWord{doc, w}]
			for i := 0; i < count; i++ {
				// sample word topic
				k := toTopic[i]

				// update word topic count
				this.WordTopicCount[w][k]++

				// update doc topic count
				this.DocTopicCount[doc][k]++

				// update topic count sum
				this.TopicCountSum[k]++
			}
		}
	}
}

// =================================================================================================
// func (this *LDA) Init
func (this *LDA) Init() {
	// ---------------------------------------------------------------------------------------------
	// step 1: randomly assign topic to word
	for doc, wordCounts := range this.Data.Docs {
		for w, count := range wordCounts {
			toTopic := this.DocWordToTopic[DocWord{doc, w}]
			for i := 0; i < count; i++ {
				// sample word topic
				k := rand.Intn(this.NumTopics)

				// record the topic assignment
				toTopic[i] = k
			}
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: update counters
	this.updateCounters()
}

// =================================================================================================
// func (this *LDA) probTopicOfDocWord
func (this *LDA) probTopicOfDocWord(doc int, w int, k int, idxK int) float64 {
	numWords := float64(len(this.Data.Vocab))
	myDocTopicCount := float64(this.DocTopicCount[doc][idxK])
	myWordTopicCount := float64(this.WordTopicCount[w][idxK])
	myTopicCountSum := float64(this.TopicCountSum[idxK])
	if idxK == k {
		myDocTopicCount--
		myWordTopicCount--
		myTopicCountSum--
	}

	docPart := this.Alpha + myDocTopicCount
	wordPart := (this.Beta + myWordTopicCount) / (this.Beta*numWords + myTopicCountSum)
	return docPart * wordPart
}

// =================================================================================================
// func (this *LDA) ResampleTopics
func (this *LDA) ResampleTopics(numIters int) {
	// ---------------------------------------------------------------------------------------------
	// step 1: resample topics in parallel
	numCPUs := runtime.NumCPU()
	for idxIter := 0; idxIter < numIters; idxIter++ {
		log.Printf("iter %5d, relative entropy %f", idxIter, this.ComputeRelativeEntropy())

		// (1.1) create channels for input and output
		chDocs := make(chan []int)
		chWorkers := make(chan bool)

		// (1.2) create goroutines for resampling
		for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
			go func() {
				prefixSum := make([]float64, this.NumTopics)
				for docs := range chDocs {
					for doc := range docs {
						wordCounts := this.Data.Docs[doc]
						for w, count := range wordCounts {
							docWord := DocWord{doc, w}
							toTopic := this.DocWordToTopic[docWord]
							for i := 0; i < count; i++ {
								k := toTopic[i]

								// (1.2.1) compute resampling probabilities
								for idxK := 0; idxK < this.NumTopics; idxK += 1 {
									prob := this.probTopicOfDocWord(doc, w, k, idxK)
									if idxK == 0 {
										prefixSum[idxK] = prob
									} else {
										prefixSum[idxK] = prefixSum[idxK-1] + prob
									}
								}

								// (1.2.2) use resampling probabilities to resample topic
								u := rand.Float64() * prefixSum[this.NumTopics-1]
								for idxK := 0; idxK < this.NumTopics; idxK++ {
									if u < prefixSum[idxK] {
										k = idxK
										break
									}
								}

								// (1.2.3) record the new topic
								toTopic[i] = k
							}
						}
					}
				}
				chWorkers <- true
			}()
		}

		// (1.3) generate inputs
		numDocs := len(this.Data.Docs)
		for doc := 0; doc < numDocs; doc += 100 {
			lenDocs := 100
			if doc+lenDocs > numDocs {
				lenDocs = numDocs - doc
			}
			docs := make([]int, lenDocs)
			for i := 0; i < lenDocs; i++ {
				docs[i] = doc + i
			}
			chDocs <- docs
		}
		close(chDocs)

		// (1.4) wait for all workers to complete their jobs
		for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
			<-chWorkers
		}

		// (1.5) update the counters
		this.updateCounters()
	}

	// step 2: report entropy
	log.Printf("final relative entropy %f", this.ComputeRelativeEntropy())
}

// =================================================================================================
// func (this *LDA) Train
// brief description: train model
func (this *LDA) Train(numIters int) {
	// randomly init
	this.Init()

	// resample topics
	this.ResampleTopics(numIters)
}

// =================================================================================================
// func (this *LDA) Infer
// brief description: infer topics on new documents
func (this *LDA) Infer(wordCounts map[int]int) []float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: compute unscaled probabilities
	probs := make([]float64, this.NumTopics)
	sumProbs := float64(0.0)
	numWords := len(this.Data.Vocab)
	for idxK := 0; idxK < this.NumTopics; idxK++ {
		myProb := float64(0.0)
		for w, count := range wordCounts {
			myWordTopicCount := float64(this.WordTopicCount[w][idxK])
			myTopicCountSum := float64(this.TopicCountSum[idxK])
			myProb += float64(count) * (this.Beta + myWordTopicCount) /
				(this.Beta*float64(numWords) + myTopicCountSum)
		}
		sumProbs += myProb
		probs[idxK] = myProb
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: scale the probs to make their sum == 1.0
	if sumProbs == 0.0 {
		sumProbs = 1.0
	}
	for idxK := 0; idxK < this.NumTopics; idxK++ {
		probs[idxK] /= sumProbs
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: return the result
	return probs
}

// =================================================================================================
// func (this *LDA) ComputeEntropy
// brief description: compute entropy
func (this *LDA) ComputeEntropy() float64 {
	entropy := 0.0
	sumCount := 0
	for doc, _ := range this.Data.Docs {
		myTopicCount := this.DocTopicCount[doc]
		mySumCount := 0
		for k := 0; k < this.NumTopics; k++ {
			mySumCount += myTopicCount[k]
		}
		myEntropy := 0.0
		for k := 0; k < this.NumTopics; k++ {
			if myTopicCount[k] > 0 {
				myP := float64(myTopicCount[k]) / float64(mySumCount)
				myEntropy -= myP * math.Log(myP)
			}
		}
		sumCount += mySumCount
		entropy += float64(mySumCount) * myEntropy
	}
	entropy /= float64(sumCount)
	return entropy
}

// =================================================================================================
// func (this *LDA) ComputeEntropy
// brief description: compute entropy
func (this *LDA) ComputeRelativeEntropy() float64 {
	entropy := this.ComputeEntropy()
	maxEntropy := -math.Log(1.0 / float64(this.NumTopics))
	return entropy / maxEntropy
}

// =================================================================================================
// function convertUnicodeHex
// brief description:
// 	Convert a hex representation string of unicode to the unicode.
// input:
//	s: The hex representation.
// output:
//	The unicode.
func convertUnicodeHex(s []byte) []byte {
	ss := s[3 : len(s)-1]
	u, err := strconv.ParseInt(string(ss), 16, 64)
	if err != nil {
		return []byte("<?>")
	} else {
		return []byte(fmt.Sprintf("%c", u))
	}
}

// =================================================================================================
// function convertUnicodeDec
// brief description:
// 	Convert a dec representation string of unicode to the unicode.
// input:
//	s: The dec representation.
// output:
//	The unicode.
func convertUnicodeDec(s []byte) []byte {
	ss := s[2 : len(s)-1]
	u, err := strconv.ParseInt(string(ss), 10, 64)
	if err != nil {
		return []byte("<?>")
	} else {
		return []byte(fmt.Sprintf("%c", u))
	}
}

// =================================================================================================
// function TidyTitle
// brief description: Make a title more tidy before using it. The procedure include the
//                    steps:
//                    (1) remove the spaces at the head and the tail of the title,
//                    (2) replace "&lt;" with "<",
//                    (3) replace "&gt;" with ">",
//                    (4) replace "&amp;" with "&",
//                    (5) replace "&quot;" with "\"",
//                    (6) replace "&apos;" with "'",
//                    (7) replace "&//[number];" with a corresponding unicode
// input:
//   title: The String (text) of a title.
// output:
//   A string of the tidied-up title.
func TidyTitle(title string) string {
	// --------------------------------------------------------------------------------------
	// step 1: remove the spaces at the head and the tail
	result := strings.TrimSpace(title)

	// --------------------------------------------------------------------------------------
	// step 2: replace "&lt;" with "<"
	result = strings.Replace(result, "&lt;", "<", -1)

	// --------------------------------------------------------------------------------------
	// step 3: replace "&gt;" with ">"
	result = strings.Replace(result, "&gt;", ">", -1)

	// --------------------------------------------------------------------------------------
	// step 4: replace "&amp;" with "&"
	result = strings.Replace(result, "&amp;", "&", -1)

	// --------------------------------------------------------------------------------------
	// step 5: replace "&quot;" with "\""
	result = strings.Replace(result, "&quot;", "\"", -1)

	// --------------------------------------------------------------------------------------
	// step 6: replace "&apos;" with "'"
	result = strings.Replace(result, "&apos;", "'", -1)

	// --------------------------------------------------------------------------------------
	// step 7: replace "&//[number];" with a corresponding unicode
	byteResult := []byte(result)
	byteResult = reUnicodeHex.ReplaceAllFunc(byteResult, convertUnicodeHex)
	byteResult = reUnicodeDec.ReplaceAllFunc(byteResult, convertUnicodeDec)
	result = string(byteResult)

	// --------------------------------------------------------------------------------------
	// step 8: return the result
	return result
}

// =================================================================================================
// function LoadCitationGraph
// brief description:
//	Load data from three files (nodes, edges, labels) of a citation graph.
// input:
//   path: The name of the path where the three files are stored.
//   prefix: The prefix of the names of the three files. For example, the prefix of
//           ijcai-citation-graph-nodes.csv is ijcai.
// output:
//    The citation graph represented by the three files.
func LoadCitationGraph(path string, prefix string) *CitationGraph {
	// --------------------------------------------------------------------------------------
	// step 1: Prepare the data structure of the result.
	result := new(CitationGraph)
	result.Nodes = make(map[int64]*CitationNode)
	result.idxMainNodes = map[int64]int{}

	// --------------------------------------------------------------------------------------
	// step 2: Assemble the file names for nodes, edges, and labels.
	fileNameOfNodes := path + "/" + prefix + "-citation-graph-nodes.csv"
	fileNameOfEdges := path + "/" + prefix + "-citation-graph-edges.csv"
	fileNameOfLabels := path + "/" + prefix + "-citation-graph-labels.csv"

	// --------------------------------------------------------------------------------------
	// step 3: Load the nodes.
	// (3.1) open the file of Nodes
	fileOfNodes, err := os.Open(fileNameOfNodes)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfNodes.Close()
	scannerOfNodes := bufio.NewScanner(fileOfNodes)

	// (3.2) Examine the first line to check whether the file format is correct.
	if !scannerOfNodes.Scan() {
		log.Fatal("Cannot read " + fileNameOfNodes)
	}
	firstLine := scannerOfNodes.Text()
	columnNames := strings.Split(firstLine, ",")
	if len(columnNames) != 4 {
		log.Fatal("Incorrect file format of " + fileNameOfNodes)
	}
	if strings.TrimSpace(columnNames[0]) != "#id" ||
		strings.TrimSpace(columnNames[1]) != "in-"+prefix ||
		strings.TrimSpace(columnNames[2]) != "year" ||
		strings.TrimSpace(columnNames[3]) != "title" {
		log.Fatal("Incorrect file format of " + fileNameOfNodes)
	}

	// (3.3) Read the rest of lines.
	for scannerOfNodes.Scan() {
		line := scannerOfNodes.Text()
		columns := strings.Split(line, ",")
		for i := 1; i < 4; i++ {
			columns[i] = strings.TrimSpace(columns[i])
		}
		id, _ := strconv.ParseInt(columns[0], 10, 64)
		isMainNode, _ := strconv.ParseBool(columns[1])
		year, _ := strconv.ParseInt(columns[2], 10, 64)
		title := strings.ReplaceAll(columns[3], "[comma]", ",")
		node := new(CitationNode)
		node.ID = id
		node.Year = year
		node.Title = title
		result.Nodes[id] = node
		if isMainNode {
			result.idxMainNodes[id] = len(result.ToBeAnalyzed)
			result.ToBeAnalyzed = append(result.ToBeAnalyzed, id)
		}
	}

	// --------------------------------------------------------------------------------------
	// step 4: Load the edges.
	// (4.1) Open the file of edges
	fileOfEdges, err := os.Open(fileNameOfEdges)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfEdges.Close()
	scannerOfEdges := bufio.NewScanner(fileOfEdges)

	// (4.2) Examine the first line to check whether the file format is correct.
	if !scannerOfEdges.Scan() {
		log.Fatal("Cannot read " + fileNameOfEdges)
	}
	firstLine = scannerOfEdges.Text()
	columnNames = strings.Split(firstLine, ",")
	if len(columnNames) != 2 {
		log.Fatal("Incorrect file format of " + fileNameOfEdges)
	}
	if strings.TrimSpace(columnNames[0]) != "#id" ||
		strings.TrimSpace(columnNames[1]) != "ref-id" {
		log.Fatal("Incorrect file format of " + fileNameOfEdges)
	}

	// (4.3) read the rest of lines
	for scannerOfEdges.Scan() {
		line := scannerOfEdges.Text()
		columns := strings.Split(line, ",")
		for i := 1; i < 2; i++ {
			columns[i] = strings.TrimSpace(columns[i])
		}
		id, _ := strconv.ParseInt(columns[0], 10, 64)
		refID, _ := strconv.ParseInt(columns[1], 10, 64)
		node, _ := result.Nodes[id]
		refNode, _ := result.Nodes[refID]
		node.Refs = append(node.Refs, refID)
		refNode.Cites = append(refNode.Cites, id)
	}

	// --------------------------------------------------------------------------------------
	// step 5: Load the labels.
	// (5.1) Open the file of labels
	fileOfLabels, err := os.Open(fileNameOfLabels)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfLabels.Close()
	scannerOfLabels := bufio.NewScanner(fileOfLabels)

	// (5.2) Examine the first line to check whether the file format is correct.
	if !scannerOfLabels.Scan() {
		log.Fatal("Cannot read " + fileNameOfLabels)
	}
	firstLine = scannerOfLabels.Text()
	columnNames = strings.Split(firstLine, ",")
	if len(columnNames) != 2 {
		log.Fatal("Incorrect file format of " + fileNameOfLabels)
	}
	if strings.TrimSpace(columnNames[0]) != "#id" ||
		strings.TrimSpace(columnNames[1]) != "label" {
		log.Fatal("Incorrect file format of " + fileNameOfEdges)
	}

	// (5.3) Read the rest of lines.
	for scannerOfLabels.Scan() {
		line := scannerOfLabels.Text()
		columns := strings.Split(line, ",")
		for i := 1; i < 2; i++ {
			columns[i] = strings.TrimSpace(columns[i])
		}
		id, _ := strconv.ParseInt(columns[0], 10, 64)
		label := strings.TrimSpace(columns[1])
		node, _ := result.Nodes[id]
		node.Labels = append(node.Labels, label)
	}

	// --------------------------------------------------------------------------------------
	// step 6: Return the result
	return result
}

// =================================================================================================
// struct GSDMM
// brief description: the data structure of GSDMM model
type GSDMM struct {
	Alpha     float64 // document topic mixture hyperparameter
	Beta      float64 // topic word mixture hyperparameter
	NumTopics int     // number of topics

	Data          *Corpus // the input corpus
	NumWordsInDoc []int   // the number of words in documents

	DocTopic          []int   // doc-topic count table
	TopicWordCount    [][]int // word-topic count table
	TopicWordCountSum []int   // word-topic-sum count table
	TopicDocCount     []int   // topic-doc-sum count table
}

// =================================================================================================
// func NewGSDMM
// brief description: create an LDA instance with collapsed gibbs sampler
func NewGSDMM(numTopics int, alpha float64, beta float64, data *Corpus) *GSDMM {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	if data == nil {
		log.Fatalln("corpus is nil")
	}

	if numTopics <= 0 {
		log.Fatalln("numTopics cannot <= 0.")
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: create docTopic
	numDocs := len(data.Docs)
	docTopic := make([]int, numDocs)

	// ---------------------------------------------------------------------------------------------
	// step 3: create topicWordCount
	numWords := len(data.Vocab)
	topicWordCount := make([][]int, numTopics)
	for k := 0; k < numTopics; k++ {
		topicWordCount[k] = make([]int, numWords)
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: create topicWordCountSum
	topicWordCountSum := make([]int, numTopics)

	// ---------------------------------------------------------------------------------------------
	// step 5: create topicDocCount
	topicDocCount := make([]int, numTopics)

	// ---------------------------------------------------------------------------------------------
	// step 6: create NumWordsInDoc
	numWordsInDoc := make([]int, numDocs)
	for doc, wordCounts := range data.Docs {
		numWordsInDoc[doc] = 0
		for _, count := range wordCounts {
			numWordsInDoc[doc] += count
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 7: assemble the result
	result := &GSDMM{
		Alpha:         alpha,
		Beta:          beta,
		NumTopics:     numTopics,
		Data:          data,
		NumWordsInDoc: numWordsInDoc,

		DocTopic:          docTopic,
		TopicWordCount:    topicWordCount,
		TopicWordCountSum: topicWordCountSum,
		TopicDocCount:     topicDocCount,
	}

	// ---------------------------------------------------------------------------------------------
	// step 8: return the result
	return result
}

// =================================================================================================
// func (this *GSDMM) updateCounters
func (this *GSDMM) updateCounters() {
	// ---------------------------------------------------------------------------------------------
	// step 1: initialize Counters
	numTopics := this.NumTopics
	numWords := len(this.Data.Vocab)
	for idxTopic := 0; idxTopic < numTopics; idxTopic++ {
		this.TopicWordCountSum[idxTopic] = 0
		this.TopicDocCount[idxTopic] = 0
		wordCountOfTopic := this.TopicWordCount[idxTopic]
		for w := 0; w < numWords; w++ {
			wordCountOfTopic[w] = 0
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: count them!
	for doc, wordCounts := range this.Data.Docs {
		k := this.DocTopic[doc]
		this.TopicDocCount[k]++
		for w, count := range wordCounts {
			this.TopicWordCount[k][w] += count
			this.TopicWordCountSum[k] += count
		}
	}
}

// =================================================================================================
// func (this *GSDMM) Init
func (this *GSDMM) Init() {
	// ---------------------------------------------------------------------------------------------
	// step 1: randomly assign topic to doc
	for doc, _ := range this.Data.Docs {
		// sample document topic
		this.DocTopic[doc] = rand.Intn(this.NumTopics)
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: update counters
	this.updateCounters()
}

// =================================================================================================
// func (this *GSDMM) probTopicOfDoc
func (this *GSDMM) probTopicOfDoc(doc int, k int, idxK int) float64 {
	numTopics := float64(this.NumTopics)
	numDocs := float64(len(this.Data.Docs))
	docCountOfTopic := float64(this.TopicDocCount[idxK])
	if idxK == k {
		docCountOfTopic--
	}
	docPart := (docCountOfTopic + this.Alpha) / (numDocs - 1.0 + this.Alpha*numTopics)

	topicWordCountSum := float64(this.TopicWordCountSum[idxK])
	wordCounts := this.Data.Docs[doc]
	if idxK == k {
		topicWordCountSum -= float64(this.NumWordsInDoc[doc])
	}

	numWords := float64(len(this.Data.Vocab))
	idxWordInDoc := 0
	wordPart := 1.0
	for w, count := range wordCounts {
		wordCountOfTopic := float64(this.TopicWordCount[idxK][w])
		if idxK == k {
			wordCountOfTopic -= float64(count)
		}
		for j := 0; j < count; j++ {
			wordPart *= (wordCountOfTopic + this.Beta + float64(j)) / (topicWordCountSum +
				this.Beta*numWords + float64(idxWordInDoc))
			idxWordInDoc++
		}
	}

	//fmt.Printf("docPart = %f, wordPart = %v, prod = %v\n", docPart, wordPart, docPart*wordPart)
	return docPart * wordPart
}

// =================================================================================================
// func (this *GSDMM) ResampleTopics
func (this *GSDMM) ResampleTopics(numIters int) {
	// ---------------------------------------------------------------------------------------------
	// step 1: resample topics in parallel
	numCPUs := runtime.NumCPU()
	for idxIter := 0; idxIter < numIters; idxIter++ {
		log.Printf("iter %5d, relative entropy %f", idxIter, this.ComputeRelativeEntropy())

		// (1.1) create channels for input and output
		chDocs := make(chan []int)
		chWorkers := make(chan bool)

		// (1.2) create goroutines for resampling
		for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
			go func() {
				prefixSum := make([]float64, this.NumTopics)
				for docs := range chDocs {
					//fmt.Println("-------------------------------------------")
					for _, doc := range docs {
						// (1.2.1) compute resampling probabilities
						k := this.DocTopic[doc]
						for idxK := 0; idxK < this.NumTopics; idxK++ {
							prob := this.probTopicOfDoc(doc, k, idxK)
							if idxK == 0 {
								prefixSum[idxK] = prob
							} else {
								prefixSum[idxK] = prefixSum[idxK-1] + prob
							}
						}
						// if doc < 10 {
						// 	fmt.Printf("%d: %v\n", doc, prefixSum)
						// }

						// (1.2.2) use resampling probabilities to resample topic
						u := rand.Float64() * prefixSum[this.NumTopics-1]
						for idxK := 0; idxK < this.NumTopics; idxK++ {
							if u < prefixSum[idxK] {
								k = idxK
								break
							}
						}

						// (1.2.3) record the new topic
						this.DocTopic[doc] = k
					}
				}
				chWorkers <- true
			}()
		}

		// (1.3) generate inputs
		numDocs := len(this.Data.Docs)
		for doc := 0; doc < numDocs; doc += 100 {
			lenDocs := 100
			if doc+lenDocs > numDocs {
				lenDocs = numDocs - doc
			}
			docs := make([]int, lenDocs)
			for i := 0; i < lenDocs; i++ {
				docs[i] = doc + i
			}
			chDocs <- docs
		}
		close(chDocs)

		// (1.4) wait for all workers to complete their jobs
		for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
			<-chWorkers
		}

		// (1.5) update the counters
		this.updateCounters()
	}

	// step 2: report entropy
	log.Printf("final relative entropy %f", this.ComputeRelativeEntropy())
}

// =================================================================================================
// func (this *GSDMM) Train
// brief description: train model
func (this *GSDMM) Train(numIters int) {
	// randomly init
	this.Init()

	// resample topics
	this.ResampleTopics(numIters)
}

// =================================================================================================
// func (this *GSDMM) Infer
// brief description: infer topics on new documents
func (this *GSDMM) Infer(wordCounts map[int]int) []float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: compute unscaled probabilities
	probs := make([]float64, this.NumTopics)
	sumProbs := float64(0.0)
	numTopics := float64(len(this.Data.Docs))
	numWords := float64(len(this.Data.Vocab))
	for idxK := 0; idxK < this.NumTopics; idxK++ {
		// -----------------------------------------------------------------------------------------
		// (1.1) compute doc part of probs
		docCountOfTopic := float64(this.TopicDocCount[idxK])
		docPart := (docCountOfTopic + this.Alpha) / (docCountOfTopic - 1.0 + this.Alpha*numTopics)

		// -----------------------------------------------------------------------------------------
		// (1.2) compute word part of probs
		topicWordCountSum := float64(this.TopicWordCountSum[idxK])
		idxWordInDoc := 0
		wordPart := 1.0
		for w, count := range wordCounts {
			wordCountOfTopic := float64(this.TopicWordCount[idxK][w])
			for j := 0; j < count; j++ {
				wordPart *= (wordCountOfTopic + this.Beta + float64(j)) / (topicWordCountSum +
					this.Beta*numWords + float64(idxWordInDoc))
				idxWordInDoc++
			}
		}

		// -----------------------------------------------------------------------------------------
		// (1.3) assemble myProb
		myProb := docPart * wordPart
		sumProbs += myProb
		probs[idxK] = myProb
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: scale the probs to make their sum == 1.0
	if sumProbs == 0.0 {
		sumProbs = 1.0
	}
	for idxK := 0; idxK < this.NumTopics; idxK++ {
		probs[idxK] /= sumProbs
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: return the result
	return probs
}

// =================================================================================================
// func (this *GSDMM) ComputeEntropy
// brief description: compute entropy
func (this *GSDMM) ComputeEntropy() float64 {
	entropy := 0.0
	numDocs := float64(len(this.Data.Docs))
	for doc, _ := range this.Data.Docs {
		k := this.DocTopic[doc]

		myProbs := make([]float64, this.NumTopics)
		mySumProbs := 0.0
		for idxK := 0; idxK < this.NumTopics; idxK++ {
			myProbs[idxK] = this.probTopicOfDoc(doc, k, idxK)
			mySumProbs += myProbs[idxK]
		}

		myEntropy := 0.0
		if mySumProbs > 0.0 {
			for idxK := 0; idxK < this.NumTopics; idxK++ {
				myP := myProbs[idxK] / mySumProbs
				if myP == 0.0 {
					continue
				}
				//fmt.Printf("%d: myP = %f\n", idxK, myP)
				myEntropy -= myP * math.Log(myP)
			}
		}

		entropy += myEntropy / numDocs
	}
	return entropy
}

// =================================================================================================
// func (this *GSDMM) ComputeEntropy
// brief description: compute entropy
func (this *GSDMM) ComputeRelativeEntropy() float64 {
	entropy := this.ComputeEntropy()
	maxEntropy := -math.Log(1.0 / float64(this.NumTopics))
	return entropy / maxEntropy
}

// =================================================================================================
// struct WPDM
// brief description: the data structure of WPDM model
type WPDM struct {
	eps    float64 // radius of neighborhood
	minPts uint    // minimum points for a dense neighborhood in DBScan, set this to 0 for AHC

	Data      *Corpus // the input corpus
	DocTopic  []int   // doc-topic table
	NumTopics int     // number of topics
}

// ===========================================================================================
// function SaveCitationGraph
// brief description: Save data to three files (nodes, edges, labels) of a citation graph.
// input:
//   path: The name of the path where the three files are stored.
//   prefix: The prefix of the names of the three files. For example, the prefix of
//           ijcai-citation-graph-nodes.csv is ijcai.
//   citationGraph: The citation graph represented by the three files.
// output:
//   nothing
func SaveCitationGraph(path string, prefix string, citationGraph *CitationGraph) {
	// --------------------------------------------------------------------------------------
	// step 1: Assemble the file names for nodes, edges, and labels.
	fileNameOfNodes := path + "/" + prefix + "-citation-graph-nodes.csv"
	fileNameOfEdges := path + "/" + prefix + "-citation-graph-edges.csv"
	fileNameOfLabels := path + "/" + prefix + "-citation-graph-labels.csv"

	// --------------------------------------------------------------------------------------
	// step 2: Save the nodes.
	// (2.1) Open the file of Nodes for writing
	fileOfNodes, err := os.Create(fileNameOfNodes)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfNodes.Close()

	// (2.2) print the first line with the file format info
	_, err = fileOfNodes.WriteString("#id, in-" + prefix + ", year, title\n")
	if err != nil {
		log.Fatal(err)
	}

	// (2.3) save data into the rest of lines
	idSetForAnalysis := make(map[int64]int)
	for i, v := range citationGraph.ToBeAnalyzed {
		idSetForAnalysis[v] = i
	}
	for id, node := range citationGraph.Nodes {
		_, toBeAnalyzed := idSetForAnalysis[id]
		year := node.Year
		title := TidyTitle(strings.ReplaceAll(node.Title, ",", "[comma]"))
		_, err = fileOfNodes.WriteString(fmt.Sprintf("%d, %t, %d, %s\n", id, toBeAnalyzed, year, title))
		if err != nil {
			log.Fatal(err)
		}
	}

	// --------------------------------------------------------------------------------------
	// step 3: Save the edges.
	// (3.1) Open the file of Edges for writing
	fileOfEdges, err := os.Create(fileNameOfEdges)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfEdges.Close()

	// (3.2) print the first line with the file format info
	_, err = fileOfEdges.WriteString("#id, ref-id\n")
	if err != nil {
		log.Fatal(err)
	}

	// (3.3) save data into the rest of lines
	edgeSet := make(map[int64]map[int64]bool)
	for id, node := range citationGraph.Nodes {
		setOfID, exists := edgeSet[id]
		if !exists {
			setOfID = make(map[int64]bool)
			edgeSet[id] = setOfID
		}
		for _, refID := range node.Refs {
			setOfID[refID] = true
		}
		for _, citeID := range node.Cites {
			setOfCiteID, exists := edgeSet[citeID]
			if !exists {
				setOfCiteID = make(map[int64]bool)
				edgeSet[citeID] = setOfCiteID
			}
			setOfCiteID[id] = true
		}
	}
	for id, setOfID := range edgeSet {
		for refID, _ := range setOfID {
			fileOfEdges.WriteString(fmt.Sprintf("%d, %d\n", id, refID))
		}
	}

	// --------------------------------------------------------------------------------------
	// step 4: Save the labels.
	// (4.1) Open the file of labels for writing
	fileOfLabels, err := os.Create(fileNameOfLabels)
	if err != nil {
		log.Fatal(err)
	}
	defer fileOfLabels.Close()

	// (4.2) print the first line with the file format info
	_, err = fileOfLabels.WriteString("#id, label\n")
	if err != nil {
		log.Fatal(err)
	}

	// (4.3) save data into the rest of lines
	for id, node := range citationGraph.Nodes {
		for _, label := range node.Labels {
			fileOfLabels.WriteString(fmt.Sprintf("%d, %s\n", id, label))
		}
	}
}

// =================================================================================================
// method TFIDF
// brief description: compute the TFIDF of possible key phrases for each main node of a citationGraph
// input:
// 	nothing
// output:
//	the result of TFIDF grouped by the main nodes of the CitationGraph
func (g *CitationGraph) TFIDF() []map[string]float64 {
	candidateTFGroups := []map[string]uint{}
	phraseCandidateGroups := [][]string{}

	for _, id := range g.ToBeAnalyzed {
		// get phrase candidates
		node := g.Nodes[id]
		phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(node.Title)

		// collect a list of auxiliary phrases
		auxPhrases := []string{}
		for _, refID := range node.Refs {
			refPhrases := KeyphraseExtraction.ExtractKeyPhraseCandidates(g.Nodes[refID].Title)
			for _, phrase := range refPhrases {
				auxPhrases = append(auxPhrases, phrase)
			}
		}

		// compute TF
		candidateTFGroups = append(candidateTFGroups, KeyphraseExtraction.TF(phraseCandidates, auxPhrases))

		// append this group of phraseCandidates to the candidate groups
		phraseCandidateGroups = append(phraseCandidateGroups, phraseCandidates)
	}

	// compute IDF
	IDFs := KeyphraseExtraction.IDF(phraseCandidateGroups)

	// TFIDF = TF * IDF
	result := []map[string]float64{}
	for _, candidateTFs := range candidateTFGroups {
		groupResult := map[string]float64{}
		for text, tf := range candidateTFs {
			idf, exists := IDFs[text]
			if !exists {
				log.Fatal(fmt.Sprintf("phrase %s not found in IDFs", text))
			}
			groupResult[text] = float64(tf) * idf
		}
		result = append(result, groupResult)
	}

	// return the result
	return result
}

// =================================================================================================
// method SimTFIDF
// brief description: compute the Fuzzy TFIDF of possible key phrases for each main node of a citationGraph
// input:
// 	phraseSimilarity: similarities of phrases
// output:
//	the result of Fuzzy TFIDF grouped by the main nodes of the CitationGraph
func (g *CitationGraph) SimTFIDF(phraseSimilarity map[string]map[string]float64) []map[string]float64 {
	numMainNodes := len(g.ToBeAnalyzed)
	candidateTFGroups := make([]map[string]float64, numMainNodes)
	phraseCandidateGroups := make([][]string, numMainNodes)

	numCPUs := runtime.NumCPU()
	chI := make(chan int)
	chWorkers := make(chan bool)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			for idx0 := range chI {
				idx1 := idx0 + 100
				if idx1 > numMainNodes {
					idx1 = numMainNodes
				}
				for idxID := idx0; idxID < idx1; idxID++ {
					// get phrase candidates
					id := g.ToBeAnalyzed[idxID]
					node := g.Nodes[id]
					phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(node.Title)

					// collect a list of auxiliary phrases
					auxPhrases := []string{}
					for _, refID := range node.Refs {
						refPhrases := KeyphraseExtraction.ExtractKeyPhraseCandidates(g.Nodes[refID].Title)
						for _, phrase := range refPhrases {
							auxPhrases = append(auxPhrases, phrase)
						}
					}

					// compute Sim TF
					candidateTFGroups[idxID] = KeyphraseExtraction.SimTF(phraseCandidates, auxPhrases, phraseSimilarity)

					// insert this group of phraseCandidates to the candidate groups
					phraseCandidateGroups[idxID] = phraseCandidates
				}
			}
			chWorkers <- true
		}()
	}
	for i := 0; i < numMainNodes; i += 100 {
		chI <- i
	}
	close(chI)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}
	fmt.Println("All sim TF computed")

	// compute IDF
	IDFs := KeyphraseExtraction.IDF(phraseCandidateGroups)
	fmt.Println(("sim IDF computed"))

	// TFIDF = TF * IDF
	result := make([]map[string]float64, numMainNodes)
	for idx, candidateTFs := range candidateTFGroups {
		groupResult := map[string]float64{}
		for text, tf := range candidateTFs {
			idf, exists := IDFs[text]
			if !exists {
				log.Fatal(fmt.Sprintf("phrase %s not found in IDFs", text))
			}
			groupResult[text] = float64(tf) * idf
		}
		sortedGroupResult := KeyphraseExtraction.ArgSort(groupResult)
		selectedGroupResult := map[string]float64{}
		for i := 0; i < len(sortedGroupResult); i++ {
			phraseI := sortedGroupResult[i]
			selected := true
			for phraseJ, _ := range selectedGroupResult {
				if KeyphraseExtraction.Includes(phraseJ, phraseI) ||
					KeyphraseExtraction.Includes(phraseI, phraseJ) {
					selected = false
					break
				}
			}
			if selected {
				selectedGroupResult[phraseI] = groupResult[phraseI]
			}
		}
		result[idx] = selectedGroupResult
	}

	// return the result
	return result
}

// =================================================================================================
// method SimTFSimIDF
// brief description: compute the Fuzzy TFIDF of possible key phrases for each main node of a citationGraph
// input:
// 	nothing
// output:
//	the result of Fuzzy TFIDF grouped by the main nodes of the CitationGraph
func (g *CitationGraph) SimTFSimIDF(phraseSimilarity map[string]map[string]float64) []map[string]float64 {
	candidateTFGroups := []map[string]float64{}
	phraseCandidateGroups := [][]string{}

	for idxID, id := range g.ToBeAnalyzed {
		// get phrase candidates
		node := g.Nodes[id]
		phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(node.Title)

		// collect a list of auxiliary phrases
		auxPhrases := []string{}
		for _, refID := range node.Refs {
			refPhrases := KeyphraseExtraction.ExtractKeyPhraseCandidates(g.Nodes[refID].Title)
			for _, phrase := range refPhrases {
				auxPhrases = append(auxPhrases, phrase)
			}
		}

		// compute Fuzzy TF
		candidateTFGroups = append(candidateTFGroups, KeyphraseExtraction.SimTF(phraseCandidates, auxPhrases, phraseSimilarity))

		// append this group of phraseCandidates to the candidate groups
		phraseCandidateGroups = append(phraseCandidateGroups, phraseCandidates)

		if (idxID+1)%1000 == 0 {
			fmt.Printf("%d of %d sim TF computed\n", idxID+1, len(g.ToBeAnalyzed))
		}
	}
	fmt.Println("All sim TF computed")

	// compute IDF
	IDFs := KeyphraseExtraction.SimIDF(phraseCandidateGroups, phraseSimilarity)
	fmt.Println(("All sim IDF computed"))

	// TFIDF = TF * IDF
	result := []map[string]float64{}
	for _, candidateTFs := range candidateTFGroups {
		groupResult := map[string]float64{}
		for text, tf := range candidateTFs {
			idf, exists := IDFs[text]
			if !exists {
				log.Fatal(fmt.Sprintf("phrase %s not found in IDFs", text))
			}
			groupResult[text] = float64(tf) * idf
		}
		result = append(result, groupResult)
	}

	// return the result
	return result
}

// =================================================================================================
// method GetPhraseSimilarity
// brief description: compute the similarities between pairs of phrases and gives them in form of a
//	sparse matrix using Title Link method
// input:
// 	nothing
// output:
//	the sparse matrix of phrase similarities
// note:
//	The title link method can be found in:
//	Bogomolova, A., Ryazanova, M., & Balk, I. (2021). Cluster approach to analysis of publication
//	titles. In Journal of Physics: Conference Series (Vol. 1727, No. 1, p. 012016). IOP Publishing.
func (g *CitationGraph) GetPhraseSimilarity(simType int) map[string]map[string]float64 {
	// --------------------------------------------------------------------------------------------
	// step 1: create corpus using all nodes
	corpus := g.CreateCorpus(2)

	// --------------------------------------------------------------------------------------------
	// step 2: get concurrences from corpuse
	concurrences := corpus.GetConcurrences()

	// --------------------------------------------------------------------------------------------
	// step 3: create concurrence model from concurrences
	cm := ConcurrenceBasedClustering.NewConcurrenceModel()
	cm.SetConcurrences(uint(len(corpus.Vocab)), concurrences)

	// --------------------------------------------------------------------------------------------
	// step 4: get similarity
	simMat := map[uint]map[uint]float64{}
	switch simType {
	case 0:
		simMat = cm.InduceSimilarities()
	case 1:
		simMat = cm.InduceNormalizedSimilarities()
	case 2:
		simMat = cm.InduceJaccardSimilarities()
	case 3:
		simMat = cm.InduceWeightedJaccardSimilarities()
	case 4:
		simMat = cm.InduceNormalizedJaccardSimilarities()
	}

	// --------------------------------------------------------------------------------------------
	// step 6: conver simMat to the result then return it
	result := map[string]map[string]float64{}
	words := make([]string, len(corpus.Vocab))
	for word, idx := range corpus.Vocab {
		words[idx] = word
	}
	for i, row := range simMat {
		wordI := words[i]
		rowI := map[string]float64{}
		for j, value := range row {
			wordJ := words[j]
			rowI[wordJ] = value
		}
		result[wordI] = rowI
	}
	return result
}

// =================================================================================================
// method GetPhraseSimilarityX
// brief description: compute the similarities between pairs of phrases and gives them in form of a
//	sparse matrix using Title Link method
// input:
// 	nothing
// output:
//	the sparse matrix of phrase similarities
// note:
//	The title link method can be found in:
//	Bogomolova, A., Ryazanova, M., & Balk, I. (2021). Cluster approach to analysis of publication
//	titles. In Journal of Physics: Conference Series (Vol. 1727, No. 1, p. 012016). IOP Publishing.
func (g *CitationGraph) GetPhraseSimilarityX(simType int) map[string]map[string]float64 {
	// --------------------------------------------------------------------------------------------
	// step 1: create corpus using all nodes
	corpus := g.CreateCorpusX(2)

	// --------------------------------------------------------------------------------------------
	// step 2: get concurrences and exclusions from corpuse
	concurrences := corpus.GetConcurrences()
	exclusions := corpus.GetExclusions()

	// --------------------------------------------------------------------------------------------
	// step 3: create concurrence model from concurrences
	cm := ConcurrenceBasedClustering.NewConcurrenceModel()
	cm.SetConcurrences(uint(len(corpus.Vocab)), concurrences)
	cm.SetExclusions(exclusions)

	// --------------------------------------------------------------------------------------------
	// step 4: get similarity
	simMat := map[uint]map[uint]float64{}
	switch simType {
	case 0:
		simMat = cm.InduceSimilarities()
	case 1:
		simMat = cm.InduceNormalizedSimilarities()
	case 2:
		simMat = cm.InduceJaccardSimilarities()
	case 3:
		simMat = cm.InduceWeightedJaccardSimilarities()
	case 4:
		simMat = cm.InduceNormalizedJaccardSimilarities()
	}

	// --------------------------------------------------------------------------------------------
	// step 6: conver simMat to the result then return it
	result := map[string]map[string]float64{}
	words := make([]string, len(corpus.Vocab))
	for word, idx := range corpus.Vocab {
		words[idx] = word
	}
	for i, row := range simMat {
		wordI := words[i]
		rowI := map[string]float64{}
		for j, value := range row {
			wordJ := words[j]
			rowI[wordJ] = value
		}
		result[wordI] = rowI
	}
	return result
}

// =================================================================================================
// func (g *CitationGraph) CreateCorpus
// brief description: create a corpus from a citation graph
// input
//	corpusType:
//		0 for title + ref titles per document for main nodes,
//		1 for title per document for main nodes,
//		2 for title per document for all nodes,
//		3 for labels per document for main nodes,
func (g *CitationGraph) CreateCorpus(corpusType int) *Corpus {
	// ---------------------------------------------------------------------------------------------
	// step 1: create an empty corpus
	corpus := NewCorpus()

	// ---------------------------------------------------------------------------------------------
	// step 2: create goroutines to add documents to the corpuse
	numCPUs := runtime.NumCPU()
	chNodes := make(chan map[int]*CitationNode)
	chWorkers := make(chan bool)
	numNodes := len(g.ToBeAnalyzed)
	if corpusType == 2 {
		numNodes = len(g.Nodes)
	}
	wordsOfNode := make([][]string, numNodes)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			for nodes := range chNodes {
				for idxNode, node := range nodes {
					// (2.1) create an empty phrase list
					words := []string{}

					// (2.2) extract words of phrases in title and put them in the phrase list
					if corpusType <= 2 {
						phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(node.Title)
						for _, candidate := range phraseCandidates {
							candidateWords := strings.Split(candidate, " ")
							for _, word := range candidateWords {
								words = append(words, word)
							}
						}
					}

					// (2.3) extract words of phrases in reference titles and put them in the
					// phrase list
					if corpusType <= 0 {
						for _, refID := range node.Refs {
							refNode := g.Nodes[refID]
							phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(refNode.Title)
							for _, candidate := range phraseCandidates {
								candidateWords := strings.Split(candidate, " ")
								for _, word := range candidateWords {
									words = append(words, word)
								}
							}
						}
					}

					// (2.4) stem labels of the node and put them in phrase list
					if corpusType == 3 {
						for _, label := range node.Labels {
							phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(label)
							for _, candidate := range phraseCandidates {
								words = append(words, candidate)
							}
						}
					}

					// (2.5) record the phrases
					wordsOfNode[idxNode] = words
				}
			}
			chWorkers <- true
		}()
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: put nodes into the input channel
	nodes := map[int]*CitationNode{}
	nodeIDsSent := map[int]bool{}
	for idx, id := range g.ToBeAnalyzed {
		// (1.1) get a node
		node := g.Nodes[id]

		// (1.2) append nodes with this node
		nodes[idx] = node
		nodeIDsSent[int(id)] = true

		// (1.3) send nodes when nodes are large enough
		if len(nodes) == 100 || idx+1 == len(g.ToBeAnalyzed) {
			chNodes <- nodes
			nodes = map[int]*CitationNode{}
		}
	}
	if corpusType == 2 {
		// send nodes not in main nodes
		idx := len(g.ToBeAnalyzed)
		for id, node := range g.Nodes {
			// (1.4) skips those already sent
			_, alreadySent := nodeIDsSent[int(id)]
			if alreadySent {
				continue
			}

			// (1.5) append nodes with this node
			nodes[idx] = node
			nodeIDsSent[int(id)] = true

			// (1.6) send nodes when nodes are large enough
			if len(nodes) == 100 {
				chNodes <- nodes
				nodes = map[int]*CitationNode{}
			}
			idx++
		}
		// (1.7) send the rest of nodes
		if len(nodes) > 0 {
			chNodes <- nodes
		}
	}
	// (1.8) close channel
	close(chNodes)

	// ---------------------------------------------------------------------------------------------
	// step 4: wait for all workers
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}
	for _, words := range wordsOfNode {
		corpus.AddDoc(words)
	}
	return corpus
}

// =================================================================================================
// func (g *CitationGraph) CreateCorpusX
// brief description: create a corpusX from a citation graph
// input
//	corpusType:
//		0 for title + ref titles per document for main nodes,
//		1 for title per document for main nodes,
//		2 for title per document for all nodes,
func (g *CitationGraph) CreateCorpusX(corpusType int) *CorpusX {
	// ---------------------------------------------------------------------------------------------
	// step 1: create an empty corpus
	corpus := NewCorpusX()

	// ---------------------------------------------------------------------------------------------
	// step 2: create goroutines to add documents to the corpuse
	numCPUs := runtime.NumCPU()
	chNodes := make(chan map[int]*CitationNode)
	chWorkers := make(chan bool)
	numNodes := len(g.ToBeAnalyzed)
	if corpusType == 2 {
		numNodes = len(g.Nodes)
	}
	wordsOfNode := make([][][]string, numNodes)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			for nodes := range chNodes {
				for idxNode, node := range nodes {
					// (2.1) create an empty phrase list
					words := [][]string{}

					// (2.2) extract words of phrases in title and put them in the phrase list
					if corpusType <= 2 {
						phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(node.Title)
						for _, candidate := range phraseCandidates {
							candidateWords := KeyphraseExtraction.GetAllPossiblePhrases(candidate)
							words = append(words, candidateWords)
						}
					}

					// (2.3) extract words of phrases in reference titles and put them in the
					// phrase list
					if corpusType <= 0 {
						for _, refID := range node.Refs {
							refNode := g.Nodes[refID]
							phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(refNode.Title)
							for _, candidate := range phraseCandidates {
								candidateWords := KeyphraseExtraction.GetAllPossiblePhrases(candidate)
								words = append(words, candidateWords)
							}
						}
					}

					// (2.4) record the phrases
					wordsOfNode[idxNode] = words
				}
			}
			chWorkers <- true
		}()
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: put nodes into the input channel
	nodes := map[int]*CitationNode{}
	nodeIDsSent := map[int]bool{}
	for idx, id := range g.ToBeAnalyzed {
		// (1.1) get a node
		node := g.Nodes[id]

		// (1.2) append nodes with this node
		nodes[idx] = node
		nodeIDsSent[int(id)] = true

		// (1.3) send nodes when nodes are large enough
		if len(nodes) == 100 || idx+1 == len(g.ToBeAnalyzed) {
			chNodes <- nodes
			nodes = map[int]*CitationNode{}
		}
	}
	if corpusType == 2 {
		// send nodes not in main nodes
		idx := len(g.ToBeAnalyzed)
		for id, node := range g.Nodes {
			// (1.4) skips those already sent
			_, alreadySent := nodeIDsSent[int(id)]
			if alreadySent {
				continue
			}

			// (1.5) append nodes with this node
			nodes[idx] = node
			nodeIDsSent[int(id)] = true

			// (1.6) send nodes when nodes are large enough
			if len(nodes) == 100 {
				chNodes <- nodes
				nodes = map[int]*CitationNode{}
			}
			idx++
		}
		// (1.7) send the rest of nodes
		if len(nodes) > 0 {
			chNodes <- nodes
		}
	}
	// (1.8) close channel
	close(chNodes)

	// ---------------------------------------------------------------------------------------------
	// step 4: wait for all workers
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}
	for _, words := range wordsOfNode {
		corpus.AddDoc(words)
	}
	return corpus
}

// =================================================================================================
// func (g *CitationGraph) GetIdxMainNode
func (g *CitationGraph) GetIdxMainNode(nodeID int64) uint {
	idx, exists := g.idxMainNodes[nodeID]
	if !exists {
		log.Fatalln(fmt.Sprintf("cannot find %v in main node indices\n", nodeID))
	}
	return uint(idx)
}

// =================================================================================================
// func (g *CitationGraph) ClusterByLDA
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using LDA.
// input:
//	numTopics: number of topics
//	alpha, beta: the parameters of LDA
//	numIters: number of iterations for LDA
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
func (g *CitationGraph) ClusterByLDA(numTopics int, alpha, beta float64, numIters int,
) map[int64][]float64 {
	if numTopics <= 0 || alpha <= 0.0 || beta <= 0.0 || numIters <= 0 { //
		log.Fatalln("all parameters of ClusterByLDA must be > 0")
	}
	// ---------------------------------------------------------------------------------------------
	// step 1: create corpus for the main nodes
	corpus := g.CreateCorpus(0)

	// ---------------------------------------------------------------------------------------------
	// step 2: use the corpus to compute LDA
	LDA := NewLDA(numTopics, alpha, beta, corpus)
	LDA.Train(numIters)

	// ---------------------------------------------------------------------------------------------
	// step 3: infer cluster memberships for each node
	memberships := map[int64][]float64{}
	for docID, doc := range corpus.Docs {
		nodeID := g.ToBeAnalyzed[docID]
		membershipsOfNode := LDA.Infer(doc)
		memberships[nodeID] = membershipsOfNode
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: return the result
	return memberships
}

// =================================================================================================
// func (g *CitationGraph) ClusterTitlesByWPDM
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using LDA.
// input:
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
//	simType: the type of similarity, 0 for simple induced similarity, 1 for normalized
//		similarity, 2 for jaccard similarity, 4 for weighted jaccard similarity, 4 for
//		normalized jaccard similarity
//	workSpaceFileName: a file name for intermediate result
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
func (g *CitationGraph) ClusterTitlesByWPDM(eps float64, minPts uint, simType int,
	workSpaceFileName string) []map[uint]bool {
	// ---------------------------------------------------------------------------------------------
	// step 1: create corpus for all nodes
	corpus := g.CreateCorpus(2)

	// ---------------------------------------------------------------------------------------------
	// step 2: create concurrences from corpus
	concurrences := corpus.GetConcurrences()

	// ---------------------------------------------------------------------------------------------
	// step 3: create corpus for main nodes
	mainCorpus := g.CreateCorpus(1)

	// ---------------------------------------------------------------------------------------------
	// step 4: translate mainCorpus to corpus
	mainCorpus = corpus.translate(mainCorpus)

	// ---------------------------------------------------------------------------------------------
	// step 5: cluster mainCorpus using WPDM
	cm := ConcurrenceBasedClustering.NewConcurrenceModel()
	cm.SetPairFilter(0.1, 3.0)
	cm.SetConcurrences(uint(len(corpus.Vocab)), concurrences)
	groups := make([]map[uint]bool, len(mainCorpus.Docs))
	for docID, doc := range mainCorpus.Docs {
		group := map[uint]bool{}
		for w, _ := range doc {
			group[uint(w)] = true
		}
		groups[docID] = group
	}
	communities := []map[uint]bool{}
	if minPts > 0 {
		communities = cm.GroupPairDBScan(groups, eps, minPts, simType, workSpaceFileName)
	} else {
		communities = cm.GroupPairAHC(groups, eps, simType, workSpaceFileName)
	}

	// ---------------------------------------------------------------------------------------------
	// step 6: return the result
	return communities
}

// =================================================================================================
// func (g *CitationGraph) ClusterLabelsByWPDM
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using LDA.
// input:
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
//	simType: the type of similarity, 0 for simple induced similarity, 1 for normalized
//		similarity, 2 for jaccard similarity, 4 for weighted jaccard similarity, 4 for
//		normalized jaccard similarity
//	workSpaceFileName: a file name for intermediate result
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
func (g *CitationGraph) ClusterLabelsByWPDM(eps float64, minPts uint, simType int,
	workSpaceFileName string) []map[uint]bool {
	// ---------------------------------------------------------------------------------------------
	// step 1: create corpus for labels at main nodes
	corpus := g.CreateCorpus(3)

	// ---------------------------------------------------------------------------------------------
	// step 2: create concurrences from corpus
	concurrences := corpus.GetConcurrences()

	// ---------------------------------------------------------------------------------------------
	// step 5: cluster corpuse using WPDM
	cm := ConcurrenceBasedClustering.NewConcurrenceModel()
	cm.SetConcurrences(uint(len(corpus.Vocab)), concurrences)
	groups := make([]map[uint]bool, len(corpus.Docs))
	for docID, doc := range corpus.Docs {
		group := map[uint]bool{}
		for w, _ := range doc {
			group[uint(w)] = true
		}
		groups[docID] = group
	}
	communities := []map[uint]bool{}
	if minPts > 0 {
		communities = cm.GroupPairDBScan(groups, eps, minPts, simType, workSpaceFileName)
	} else {
		communities = cm.GroupPairAHC(groups, eps, simType, workSpaceFileName)
	}

	// ---------------------------------------------------------------------------------------------
	// step 6: return the result
	return communities
}

// =================================================================================================
// func (g *CitationGraph) ClusterTitlesByDBScan
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using LDA.
// input:
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
//	simType: the type of similarity, 0 for simple induced similarity, 1 for normalized
//		similarity, 2 for jaccard similarity, 4 for weighted jaccard similarity, 4 for
//		normalized jaccard similarity
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
// func (g *CitationGraph) ClusterTitlesByDBScan(eps float64, minPts uint, simType int,
// ) map[int64][]float64 {

// }

// =================================================================================================
// func (g *CitationGraph) ClusterLabelsByDBScan
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using LDA.
// input:
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
//	simType: the type of similarity, 0 for simple induced similarity, 1 for normalized
//		similarity, 2 for jaccard similarity, 4 for weighted jaccard similarity, 4 for
//		normalized jaccard similarity
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
// func (g *CitationGraph) ClusterTitlesByDBScan(eps float64, minPts uint, simType int,
// ) map[int64][]float64 {

// }

// =================================================================================================
// func (g *CitationGraph) ClusterTitlesByGSDMM
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using LDA.
// input:
//	simType: the type of similarity, 0 for simple induced similarity, 1 for normalized
//		similarity, 2 for jaccard similarity, 4 for weighted jaccard similarity, 4 for
//		normalized jaccard similarity
//	numTopics: number of topics
//	alpha, beta: the parameters of GSDMM
//	numIters: number of iterations
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
func (g *CitationGraph) ClusterTitlesByGSDMM(simType, numTopics int, alpha, beta float64, numIters int,
) map[int64][]float64 {
	if numTopics <= 0 || alpha <= 0.0 || beta <= 0.0 || numIters <= 0 {
		log.Fatalln("all parameters of ClusterByLDA must be > 0")
	}
	// ---------------------------------------------------------------------------------------------
	// step 1: create corpus for the main nodes
	phraseSimilarities := g.GetPhraseSimilarityX(simType)
	tfidfGroups := g.SimTFIDF(phraseSimilarities)
	corpus := NewCorpus()
	for _, tfidfs := range tfidfGroups {
		texts := []string{}
		sumWeights := 0.0
		for _, weight := range tfidfs {
			sumWeights += weight
		}
		meanWeight := sumWeights / float64(len(tfidfs))
		for text, weight := range tfidfs {
			if weight < 0.5*meanWeight {
				continue
			}
			texts = append(texts, text)
		}
		corpus.AddDoc(texts)
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: use the corpus to compute LDA
	GSDMM := NewGSDMM(numTopics, alpha, beta, corpus)
	GSDMM.Train(numIters)

	// ---------------------------------------------------------------------------------------------
	// step 3: infer cluster memberships for each node
	memberships := map[int64][]float64{}
	for docID, doc := range corpus.Docs {
		nodeID := g.ToBeAnalyzed[docID]
		membershipsOfNode := GSDMM.Infer(doc)
		memberships[nodeID] = membershipsOfNode
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: return the result
	return memberships
}

// =================================================================================================
// func (g *CitationGraph) ClusterLabelsByGSDMM
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using LDA.
// input:
//	eps: the radius of neighborhood.
//	minPts: Only if the neighborhood of a point contains at least minPt points
//		(the center point of the neighborhood included), the neighborhood is
//		called dense. Only dense neighborhoods are connected to communities.
//	simType: the type of similarity, 0 for simple induced similarity, 1 for normalized
//		similarity, 2 for jaccard similarity, 4 for weighted jaccard similarity, 4 for
//		normalized jaccard similarity
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
// func (g *CitationGraph) ClusterTitlesByGSDMM(eps float64, minPts uint, simType int,
// ) map[int64][]float64 {

// }
func (g *CitationGraph) ClusterLabelsByGSDMM(numTopics int, alpha, beta float64, numIters int,
) map[int64][]float64 {
	if numTopics <= 0 || alpha <= 0.0 || beta <= 0.0 || numIters <= 0 {
		log.Fatalln("all parameters of ClusterByLDA must be > 0")
	}
	// ---------------------------------------------------------------------------------------------
	// step 1: create corpus for labels of the main nodes
	corpus := g.CreateCorpus(3)
	// reverse corpus Vocab
	reverseVocab := map[int]string{}
	for w, j := range corpus.Vocab {
		reverseVocab[j] = w
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: use the corpus to compute LDA
	GSDMM := NewGSDMM(numTopics, alpha, beta, corpus)
	GSDMM.Train(numIters)

	// ---------------------------------------------------------------------------------------------
	// step 3: infer cluster memberships for each node
	memberships := map[int64][]float64{}
	for docID, doc := range corpus.Docs {
		nodeID := g.ToBeAnalyzed[docID]
		membershipsOfNode := GSDMM.Infer(doc)
		memberships[nodeID] = membershipsOfNode
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: return the result
	return memberships
}

// =================================================================================================
// func (g *CitationGraph) checkMemberships
func (g *CitationGraph) checkMemberships(memberships map[int64][]float64) {
	if len(memberships) != len(g.ToBeAnalyzed) {
		log.Fatalln("length of memberships is incorrect.")
	}

	numDims := 0
	for _, nodeID := range g.ToBeAnalyzed {
		membership, exists := memberships[nodeID]
		if !exists {
			log.Fatalln("some nodes for analysis cannot be found in memberships")
		}
		if numDims == 0 {
			numDims = len(membership)
		} else if numDims != len(membership) {
			log.Fatalln("number of dimensions are inconsistent in memberships")
		}
	}
}

// =================================================================================================
// func (g *CitationGraph) checkCommunities
func (g *CitationGraph) checkCommunities(communities []map[uint]bool) {
	numNodes := 0
	for _, community := range communities {
		numNodes += len(community)
	}
	if numNodes != len(g.ToBeAnalyzed) {
		log.Fatalln("total number of nodes in communities is incorrect.")
	}
}

// =================================================================================================
// func membCos
func membCos(membershipI, membershipJ []float64) float64 {
	if len(membershipI) != len(membershipJ) {
		log.Fatalln("lengthes of membershipI and membershipJ don't match.")
	}
	sim := 0.0
	sizeI := 0.0
	sizeJ := 0.0
	for k, membI := range membershipI {
		membJ := membershipJ[k]
		sim += membI * membJ
		sizeI += membI * membI
		sizeJ += membJ * membJ
	}
	sizeI = math.Sqrt(sizeI)
	sizeJ = math.Sqrt(sizeJ)
	sim /= sizeI * sizeJ
	return sim
}

// =================================================================================================
// func (g *CitationGraph) CompareByModularity
func (g *CitationGraph) CompareByModularity(communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	g.checkCommunities(communities)
	g.checkMemberships(memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute sumSims
	n := len(g.ToBeAnalyzed)
	sumSims := make([]float64, n)
	numCPUs := runtime.NumCPU()
	chI := make(chan int)
	chWorkers := make(chan bool)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			for i0 := range chI {
				i1 := i0 + 100
				if i1 > n {
					i1 = n
				}
				for i := i0; i < i1; i++ {
					nodeIDOfI := g.ToBeAnalyzed[i]
					membershipI := memberships[nodeIDOfI]
					sumSims[i] = 0.0
					for _, nodeIDOfJ := range g.ToBeAnalyzed {
						membershipJ := memberships[nodeIDOfJ]
						sim := membCos(membershipI, membershipJ)
						sumSims[i] += sim
					}
				}
			}
			chWorkers <- true
		}()
	}
	for i := 0; i < n; i += 100 {
		chI <- i
	}
	close(chI)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}
	totalSumSims := 0.0
	for i := 0; i < n; i++ {
		totalSumSims += sumSims[i]
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: compute modularity
	modularity := 0.0
	for _, community := range communities {
		for i, _ := range community {
			nodeIDOfI := g.ToBeAnalyzed[i]
			membershipI := memberships[nodeIDOfI]
			for j, _ := range community {
				nodeIDOfJ := g.ToBeAnalyzed[j]
				membershipJ := memberships[nodeIDOfJ]
				sim := membCos(membershipI, membershipJ)
				modularity += sim - sumSims[i]*sumSims[j]/totalSumSims
			}
		}
	}
	modularity /= totalSumSims

	// ---------------------------------------------------------------------------------------------
	// step 4: return result
	return modularity
}

// =================================================================================================
// func (g *CitationGraph) CompareByCPM
func (g *CitationGraph) CompareByCPM(gamma float64, communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	g.checkCommunities(communities)
	g.checkMemberships(memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute CPM
	cpm := 0.0
	for _, community := range communities {
		for i, _ := range community {
			nodeIDOfI := g.ToBeAnalyzed[i]
			membershipI := memberships[nodeIDOfI]
			for j, _ := range community {
				nodeIDOfJ := g.ToBeAnalyzed[j]
				membershipJ := memberships[nodeIDOfJ]
				sim := membCos(membershipI, membershipJ)
				cpm += sim - gamma
			}
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: return result
	return cpm
}

// =================================================================================================
// func (g *CitationGraph) getCommunitiesFromMemberships
func (g *CitationGraph) GetCommunitiesFromMemberships(memberships map[int64][]float64) []map[uint]bool {
	numCommunities := 0
	for _, membership := range memberships {
		numCommunities = len(membership)
		break
	}
	result := make([]map[uint]bool, numCommunities)
	for i := 0; i < numCommunities; i++ {
		result[i] = map[uint]bool{}
	}
	for nodeID, membership := range memberships {
		maxMemb := 0.0
		idxMaxMemb := 0
		for k := 0; k < numCommunities; k++ {
			memb := membership[k]
			if memb > maxMemb {
				maxMemb = memb
				idxMaxMemb = k
			}
		}
		result[idxMaxMemb][g.GetIdxMainNode(nodeID)] = true
	}
	return result
}

// =================================================================================================
// func getCommunityIDFromCommunities
func getCommunityIDsFromCommunities(n int, communities []map[uint]bool) []int {
	result := make([]int, n)
	for i, community := range communities {
		for u, _ := range community {
			result[u] = i
		}
	}
	return result
}

// =================================================================================================
// func (g *CitationGraph) CompareByRI
func (g *CitationGraph) CompareByRI(communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	g.checkCommunities(communities)
	g.checkMemberships(memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute the intersection part
	partIntersect := 0.0
	myCommunities := g.GetCommunitiesFromMemberships(memberships)
	for _, ci := range communities {
		if len(ci) == 0 {
			continue
		}
		for _, cj := range myCommunities {
			if len(cj) == 0 {
				continue
			}
			cu, cv := ci, cj
			if len(cu) > len(cv) {
				cu, cv = cv, cu
			}
			numInBoth := 0
			for u, _ := range cu {
				_, inCv := cv[u]
				if inCv {
					numInBoth++
				}
			}
			partIntersect += float64(numInBoth * (numInBoth - 1) / 2)
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: compute the cross-boundary part
	n := len(g.ToBeAnalyzed)
	commuID1s := getCommunityIDsFromCommunities(n, communities)
	commuID2s := getCommunityIDsFromCommunities(n, myCommunities)
	partCross := 0.0
	for u := 0; u < n; u++ {
		c1u := commuID1s[u]
		c2u := commuID2s[u]
		for v := 0; v < n; v++ {
			c1v := commuID1s[v]
			c2v := commuID2s[v]
			if c1u == c1v || c2u == c2v {
				continue
			}
			partCross += 0.5
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: normalize and return the result
	partN := float64(n * (n - 1) / 2)
	fmt.Printf("part intersect = %f, partCross = %f, partN = %f\n", partIntersect, partCross, partN)
	result := (partIntersect + partCross) / partN
	return result
}

// =================================================================================================
// func (g *CitationGraph) CompareByARI
func (g *CitationGraph) CompareByARI(communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	g.checkCommunities(communities)
	g.checkMemberships(memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute the intersection part
	result := 0.0
	myCommunities := g.GetCommunitiesFromMemberships(memberships)
	for _, ci := range communities {
		if len(ci) == 0 {
			continue
		}
		for _, cj := range myCommunities {
			if len(cj) == 0 {
				continue
			}
			cu, cv := ci, cj
			if len(cu) > len(cv) {
				cu, cv = cv, cu
			}
			numInBoth := 0
			for u, _ := range cu {
				_, inCv := cv[u]
				if inCv {
					numInBoth++
				}
			}
			result += float64(numInBoth * (numInBoth - 1) / 2)
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: compute the cross-boundary part
	n := len(g.ToBeAnalyzed)
	partN := float64(n * (n - 1) / 2)
	partI := 0.0
	for _, ci := range communities {
		ni := len(ci)
		if ni <= 1 {
			continue
		}
		partI += float64(ni * (ni - 1) / 2)
	}
	partJ := 0.0
	for _, cj := range myCommunities {
		nj := len(cj)
		if nj <= 1 {
			continue
		}
		partJ += float64(nj * (nj - 1) / 2)
	}
	partCross := partI * partJ / partN

	fmt.Printf("part intersect = %f, partCross = %f, partI = %f, partJ = %f\n", result, partCross, partI, partJ)

	// ---------------------------------------------------------------------------------------------
	// step 4: normalize and return the result
	result = (result - partCross) / (0.5*(partI+partJ) - partCross)
	return result
}

// =================================================================================================
// func (g *CitationGraph) ComputeEntropies
func (g *CitationGraph) ComputeEntropies(communities []map[uint]bool,
	memberships map[int64][]float64) (float64, float64, float64) {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	g.checkCommunities(communities)
	g.checkMemberships(memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute the cross entropy
	crossEntropy := 0.0
	n := len(g.ToBeAnalyzed)
	myCommunities := g.GetCommunitiesFromMemberships(memberships)
	for _, ci := range communities {
		if len(ci) == 0 {
			continue
		}
		for _, cj := range myCommunities {
			if len(cj) == 0 {
				continue
			}
			cu, cv := ci, cj
			if len(cu) > len(cv) {
				cu, cv = cv, cu
			}
			numInBoth := 0
			for u, _ := range cu {
				_, inCv := cv[u]
				if inCv {
					numInBoth++
				}
			}
			if numInBoth == 0 {
				continue
			}
			p := float64(numInBoth) / float64(n)
			crossEntropy -= p * math.Log(p)
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: compute the entropies
	entropy1 := 0.0
	for _, ci := range communities {
		ni := len(ci)
		if ni == 0 {
			continue
		}
		p := float64(ni) / float64(n)
		entropy1 -= p * math.Log(p)
	}

	entropy2 := 0.0
	for _, cj := range myCommunities {
		nj := len(cj)
		if nj == 0 {
			continue
		}
		p := float64(nj) / float64(n)
		entropy2 -= p * math.Log(p)
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: return the entropies
	return crossEntropy, entropy1, entropy2
}

// =================================================================================================
// func (g *CitationGraph) CompareByMI
func (g *CitationGraph) CompareByMI(communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: compute the entropies
	crossEntropy, entropy1, entropy2 := g.ComputeEntropies(communities, memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute and return the result
	return entropy1 + entropy2 - crossEntropy
}

// =================================================================================================
// func (g *CitationGraph) CompareByNMI
func (g *CitationGraph) CompareByNMI(communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: compute the entropies
	crossEntropy, entropy1, entropy2 := g.ComputeEntropies(communities, memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute and return the result
	return 2.0 * (entropy1 + entropy2 - crossEntropy) / (entropy1 + entropy2)
}

func numCombs(a ...int) float64 {
	m := len(a)
	m1 := m
	for i := 0; i < m; i++ {
		if a[i] < 0 {
			m1 = i
			break
		}
	}
	if m1 == m {
		log.Fatalln(fmt.Sprintf("in numComb, m1 == m\n"))
	}

	b := make([]int, m)
	n1 := 0
	for i := 0; i < m1; i++ {
		b[i] = a[i]
		n1 += a[i]
	}
	n2 := 0
	for i := m1 + 1; i < m; i++ {
		b[i] = a[i]
		n2 += a[i]
	}

	if n1 != n2 {
		log.Fatalln(fmt.Sprintf("in numComb, n1 = %d != n2 = %d\n", n1, n2))
	}

	result := 1.0
	for k := 0; k < n1; k++ {
		i1Max := 0
		for i1 := 1; i1 < m1; i1++ {
			if b[i1] > b[i1Max] {
				i1Max = i1
			}
		}

		i2Max := m1 + 1
		for i2 := m1 + 2; i2 < m; i2++ {
			if b[i2] > b[i2Max] {
				i2Max = i2
			}
		}

		result *= float64(b[i1Max]) / float64(b[i2Max])
		b[i1Max]--
		b[i2Max]--
	}
	return result
}

// =================================================================================================
// func (g *CitationGraph) ComputeEMI
func (g *CitationGraph) ComputeEMI(communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: check parameters
	g.checkCommunities(communities)
	g.checkMemberships(memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute the cross entropy
	result := 0.0
	n := len(g.ToBeAnalyzed)
	myCommunities := g.GetCommunitiesFromMemberships(memberships)
	for _, ci := range communities {
		ni := len(ci)
		if ni == 0 {
			continue
		}
		for _, cj := range myCommunities {
			nj := len(cj)
			if nj == 0 {
				continue
			}
			k0 := ni + nj - n
			if k0 < 1 {
				k0 = 1
			}
			k1 := ni
			if k1 > nj {
				k1 = nj
			}
			for k := k0; k <= k1; k++ {
				result += numCombs(ni, nj, n-ni, n-nj, -1, n, k, ni-k, nj-k, n-ni-nj+k) *
					float64(k) / float64(n) * math.Log(float64(k*n)/float64(ni*nj))
			}
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: return the result
	return result
}

// =================================================================================================
// func (g *CitationGraph) CompareByAMI
func (g *CitationGraph) CompareByAMI(communities []map[uint]bool,
	memberships map[int64][]float64) float64 {
	// ---------------------------------------------------------------------------------------------
	// step 1: compute the entropies
	crossEntropy, entropy1, entropy2 := g.ComputeEntropies(communities, memberships)

	// ---------------------------------------------------------------------------------------------
	// step 2: compute and expected mutual entropy
	emi := g.ComputeEMI(communities, memberships)

	// ---------------------------------------------------------------------------------------------
	// step 3: compute and return the result
	mi := entropy1 + entropy2 - crossEntropy
	return (mi - emi) / (math.Max(entropy1, entropy2) - emi)
}

// =================================================================================================
// func SaveMemberships
func SaveMemberships(memberships map[int64][]float64, fileName string) {
	fileMembership, err := os.Create(fileName)
	if err != nil {
		log.Fatal(err)
	}
	jsonBytes, err := json.Marshal(memberships)
	if err != nil {
		log.Fatal(err)
	}
	jsonString := string(jsonBytes)
	_, err = fileMembership.WriteString(jsonString)
	if err != nil {
		log.Fatal(err)
	}
	fileMembership.Close()
}

// =================================================================================================
// func LoadMemberships
func LoadMemberships(fileName string) map[int64][]float64 {
	memberships := map[int64][]float64{}
	fileMembership, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	bufLen := 1024 * 1024
	bytes := make([]byte, bufLen)
	jsonBytes := []byte{}
	readLen, err := fileMembership.Read(bytes)
	if err != nil {
		log.Fatal(err)
	}
	for readLen == bufLen {
		jsonBytes = append(jsonBytes, bytes...)
		readLen, err = fileMembership.Read(bytes)
		if err != nil {
			log.Fatal(err)
		}
	}
	if readLen > 0 {
		jsonBytes = append(jsonBytes, bytes[0:readLen]...)
	}

	fileMembership.Close()
	json.Unmarshal(jsonBytes, &memberships)
	return memberships
}

// =================================================================================================
// struct PhrasePair
type PhrasePair struct {
	Phrase1, Phrase2 string
}

type PairFreq struct {
	Actual   float64
	Expected float64
}

// =================================================================================================
// func (g *CitationGraph) preparePhrasePairSearch
func (g *CitationGraph) preparePhrasePairSearch() (concurrences map[uint]map[uint]float64,
	sumConcurrencesOf []float64, phrases []string) {
	// --------------------------------------------------------------------------------------------
	// step 1: create corpus using all nodes
	corpus := g.CreateCorpusX(2)

	// --------------------------------------------------------------------------------------------
	// step 2: get concurrences and exclusions from corpuse
	concurrences = corpus.GetConcurrences()

	// --------------------------------------------------------------------------------------------
	// step 3: compute sum of concurrences
	numPhrases := uint(len(corpus.Vocab))
	sumConcurrencesOf = ConcurrenceBasedClustering.GetSumConcurrencesOf(numPhrases, concurrences)

	// --------------------------------------------------------------------------------------------
	// step 4: create concurrence model from concurrences
	phrases = make([]string, numPhrases)
	for phrase, idx := range corpus.Vocab {
		phrases[idx] = phrase
	}

	return
}

// =================================================================================================
// func (g *CitationGraph) findStronglyConnectedPhrases
func (g *CitationGraph) findStronglyConnectedPhrases(concurrences map[uint]map[uint]float64,
	sumConcurrencesOf []float64, phrases []string, thresFreq, thresRatio float64,
) map[PhrasePair]PairFreq {
	// --------------------------------------------------------------------------------------------
	// step 1: apply thresholds in filtering
	n := len(g.Nodes)
	numPhrases := len(phrases)
	result := map[PhrasePair]PairFreq{}
	for w1, concurrencesOfW1 := range concurrences {
		relFreq1 := sumConcurrencesOf[w1] / float64(numPhrases)
		for w2, freq := range concurrencesOfW1 {
			if w1 >= w2 {
				continue
			}
			if freq < thresFreq {
				continue
			}
			relFreq2 := sumConcurrencesOf[w2] / float64(numPhrases)
			expectedFreq := relFreq1 * relFreq2 * float64(n)
			if freq >= thresRatio*expectedFreq {
				result[PhrasePair{phrases[w1], phrases[w2]}] = PairFreq{Actual: freq, Expected: expectedFreq}
			}
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: return the result
	return result
}

// =================================================================================================
// func (g *CitationGraph) GetStronglyConnectedPhrases
func (g *CitationGraph) GetStronglyConnectedPhrases(thresFreq, thresRatio float64) map[PhrasePair]PairFreq {
	// --------------------------------------------------------------------------------------------
	// step 1: get concurrences, sumConcurrencesOf, and phrases
	concurrences, sumConcurrencesOf, phrases := g.preparePhrasePairSearch()

	// --------------------------------------------------------------------------------------------
	// step 2: apply thresholds in filtering and return the result
	return g.findStronglyConnectedPhrases(concurrences, sumConcurrencesOf, phrases, thresFreq, thresRatio)

}
