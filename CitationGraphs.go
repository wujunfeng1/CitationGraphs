// CitationGraphs.go project CitationGraphs.go
package CitationGraphs

import (
	"bufio"
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
}

// =================================================================================================
// struct Corpus
// brief description: the corpus data structure of LDA algorithms
type Corpus struct {
	Vocab map[string]int // the vocabulary
	Docs  []map[int]int  // keys: docID, wordID, value: word count
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
			wordID := len(this.Vocab)
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
// interface LDAModel
// brief description: the common interface of LDA models
type LDAModel interface {
	// train model for iter iteration
	Train(numIters int)
	// do inference for new doc with its wordCounts
	Infer(words []string) []float32
	// compute entropy
	ComputeEntropy() float64
}

// =================================================================================================
// struct DocWord
type DocWord struct {
	DocId  int
	WordId int
}

// =================================================================================================
// struct CGSLDA
// brief description: the data structure of LDA model with Collapsed Gibbs Sampler
// note:
//	The fast collapsed gibbs sampler algorithm can be found in reference:
//	Porteous, I., Newman, D., Ihler, A., Asuncion, A., Smyth, P., & Welling, M. (2008, August). Fast
//	collapsed gibbs sampling for latent dirichlet allocation. In Proceedings of the 14th ACM SIGKDD
//	international conference on Knowledge discovery and data mining (pp. 569-577).
type CGSLDA struct {
	Alpha     float32 // document topic mixture hyperparameter
	Beta      float32 // topic word mixture hyperparameter
	numTopics int     // number of topics

	Data *Corpus // the input corpus

	WordTopicCount [][]int           // word-topic count table
	DocTopicCount  [][]int           // doc-topic count table
	TopicCountSum  []int             // word-topic-sum count table
	DocWordToTopic map[DocWord][]int // doc-word-topic count table
}

// =================================================================================================
// func NewCGSLDA
// brief description: create an LDA instance with collapsed gibbs sampler
func NewCGSLDA(numTopics int, alpha float32, beta float32, data *Corpus) *CGSLDA {
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
	result := &CGSLDA{
		Alpha:     alpha,
		Beta:      beta,
		numTopics: numTopics,
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
// func (this *CGSLDA) updateCounters
func (this *CGSLDA) updateCounters() {
	// ---------------------------------------------------------------------------------------------
	// step 1: initialize Counters
	numTopics := this.numTopics
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
// func (this *CGSLDA) Init
func (this *CGSLDA) Init() {
	// ---------------------------------------------------------------------------------------------
	// step 1: randomly assign topic to word
	for doc, wordCounts := range this.Data.Docs {
		for w, count := range wordCounts {
			toTopic := this.DocWordToTopic[DocWord{doc, w}]
			for i := 0; i < count; i++ {
				// sample word topic
				k := rand.Intn(this.numTopics)

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
// func (this *CGSLDA) ResampleTopics
func (this *CGSLDA) ResampleTopics(numIters int) {
	// ---------------------------------------------------------------------------------------------
	// step 1: resample topics in parallel
	numCPUs := runtime.NumCPU()
	numWords := len(this.Data.Vocab)
	for idxIter := 0; idxIter < numIters; idxIter++ {
		log.Printf("iter %5d, entropy %f", idxIter, this.ComputeEntropy())

		// (1.1) create channels for input and output
		chDocs := make(chan []int)
		chWorkers := make(chan bool)

		// (1.2) create goroutines for resampling
		for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
			go func() {
				prefixSum := make([]float32, this.numTopics)
				for docs := range chDocs {
					for doc := range docs {
						wordCounts := this.Data.Docs[doc]
						for w, count := range wordCounts {
							docWord := DocWord{doc, w}
							toTopic := this.DocWordToTopic[docWord]
							for i := 0; i < count; i++ {
								k := toTopic[i]

								// (1.2.1) compute resampling probabilities
								for idxK := 0; idxK < this.numTopics; idxK += 1 {
									myDocTopicCount := float32(this.DocTopicCount[doc][idxK])
									myWordTopicCount := float32(this.WordTopicCount[w][idxK])
									myTopicCountSum := float32(this.TopicCountSum[idxK])
									if idxK == k {
										myDocTopicCount--
										myWordTopicCount--
										myTopicCountSum--
									}

									docPart := this.Alpha + myDocTopicCount
									wordPart := (this.Beta + myWordTopicCount) /
										(this.Beta*float32(numWords) + myTopicCountSum)
									if idxK == 0 {
										prefixSum[idxK] = docPart * wordPart
									} else {
										prefixSum[idxK] = prefixSum[idxK-1] + docPart*wordPart
									}
								}

								// (1.2.2) use resampling probabilities to resample topic
								u := rand.Float32() * prefixSum[this.numTopics-1]
								for idxK := 0; idxK < this.numTopics; idxK++ {
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
	log.Printf("final entropy %f", this.ComputeEntropy())
}

// =================================================================================================
// func (this *CGSLDA) Train
// brief description: train model
func (this *CGSLDA) Train(numIters int) {
	// randomly init
	this.Init()

	// resample topics
	this.ResampleTopics(numIters)
}

// =================================================================================================
// func (this *CGSLDA) Infer
// brief description: infer topics on new documents
func (this *CGSLDA) Infer(words []string) []float32 {
	// ---------------------------------------------------------------------------------------------
	// step 1: convert words to word counts
	wordCounts := map[int]int{}
	for _, word := range words {
		w, exists := this.Data.Vocab[word]
		if exists {
			count, exists := wordCounts[w]
			if !exists {
				count = 0
			}
			wordCounts[w] = count + 1
		}
	}

	// ---------------------------------------------------------------------------------------------
	// step 2: compute unscaled probabilities
	probs := make([]float32, this.numTopics)
	sumProbs := float32(0.0)
	numWords := len(this.Data.Vocab)
	for idxK := 0; idxK < this.numTopics; idxK++ {
		myProb := float32(0.0)
		for w, count := range wordCounts {
			myWordTopicCount := float32(this.WordTopicCount[w][idxK])
			myTopicCountSum := float32(this.TopicCountSum[idxK])
			myProb += float32(count) * (this.Beta + myWordTopicCount) /
				(this.Beta*float32(numWords) + myTopicCountSum)
		}
		sumProbs += myProb
		probs[idxK] = myProb
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: scale the probs to make their sum == 1.0
	if sumProbs == 0.0 {
		sumProbs = 1.0
	}
	for idxK := 0; idxK < this.numTopics; idxK++ {
		probs[idxK] /= sumProbs
	}

	// ---------------------------------------------------------------------------------------------
	// step 4: return the result
	return probs
}

// =================================================================================================
// func (this *CGSLDA) ComputeEntropy
// brief description: compute entropy
func (this *CGSLDA) ComputeEntropy() float64 {
	entropy := 0.0
	sumCount := 0
	for doc, _ := range this.Data.Docs {
		myTopicCount := this.DocTopicCount[doc]
		mySumCount := 0
		for k := 0; k < this.numTopics; k++ {
			mySumCount += myTopicCount[k]
		}
		myEntropy := 0.0
		for k := 0; k < this.numTopics; k++ {
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
	IDFs := KeyphraseExtraction.IDF(phraseCandidateGroups)
	fmt.Println(("sim IDF computed"))

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
// method GetPhraseSimilarityTL
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
func (g *CitationGraph) GetPhraseSimilarityTL() map[string]map[string]float64 {
	// --------------------------------------------------------------------------------------------
	// step 1: count the phrase-pair frequencies
	pairFreq := map[string]map[string]uint{}
	numCPUs := runtime.NumCPU()
	chNodes := make(chan []*CitationNode)
	chWorkers := make(chan bool)
	lock := sync.RWMutex{}
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func(idxWorker int) {
			// store my results in myPairFreq first, merge my results into pairFreq later
			myPairFreq := map[string]map[string]uint{}
			numProcessedNodes := 0
			for nodes := range chNodes {
				for _, node := range nodes {
					// get phrase candidates
					phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(node.Title)

					// split candidates
					wordss := [][]string{}
					for _, candid := range phraseCandidates {
						words := strings.Split(candid, " ")
						wordss = append(wordss, words)
					}

					// count pair frequency
					for idx1, words1 := range wordss {
						n1 := len(words1)
						for idx2, words2 := range wordss {
							if idx2 == idx1 {
								continue
							}
							n2 := len(words2)
							for i1 := 0; i1 < n1; i1++ {
								text1 := words1[i1]
								row, exists := myPairFreq[text1]
								if !exists {
									row = map[string]uint{}
									myPairFreq[text1] = row
								}
								for i2 := 0; i2 < n2; i2++ {
									text2 := words2[i2]
									oldFreq, exists := row[text2]
									if !exists {
										oldFreq = 0
									}
									row[text2] = oldFreq + 1
									for j2 := i2 + 1; j2 < n2; j2++ {
										text2 += " " + words2[j2]
										oldFreq, exists = row[text2]
										if !exists {
											oldFreq = 0
										}
										row[text2] = oldFreq + 1
									}
								}
								for j1 := i1 + 1; j1 < n1; j1++ {
									text1 += " " + words1[j1]
									row, exists = myPairFreq[text1]
									if !exists {
										row = map[string]uint{}
										myPairFreq[text1] = row
									}
									for i2 := 0; i2 < n2; i2++ {
										text2 := words2[i2]
										oldFreq, exists := row[text2]
										if !exists {
											oldFreq = 0
										}
										row[text2] = oldFreq + 1
										for j2 := i2 + 1; j2 < n2; j2++ {
											text2 += " " + words2[j2]
											oldFreq, exists = row[text2]
											if !exists {
												oldFreq = 0
											}
											row[text2] = oldFreq + 1
										}
									}
								}
							}
						}
					}

				}
				numProcessedNodes += len(nodes)
				fmt.Printf("worker %d has processed %d nodes\n", idxWorker, numProcessedNodes)
			}

			// now we merge the results into pairFreq
			// fmt.Printf("worker %d is merging results\n", idxWorker)
			lock.Lock()
			defer lock.Unlock()
			for key1, value1 := range myPairFreq {
				row, exists := pairFreq[key1]
				if !exists {
					pairFreq[key1] = value1
				} else {
					for key2, value2 := range value1 {
						oldValue, exists := row[key2]
						if !exists {
							oldValue = uint(0)
						}
						row[key2] = oldValue + value2
					}
				}
			}
			fmt.Printf("worker %d exits\n", idxWorker)
			chWorkers <- true
		}(idxCPU)
	}

	nodeBatch := make([]*CitationNode, 3000)
	idxInBatch := 0
	for _, node := range g.Nodes {
		nodeBatch[idxInBatch] = node
		idxInBatch++
		if idxInBatch == len(nodeBatch) {
			chNodes <- nodeBatch
			nodeBatch = make([]*CitationNode, 3000)
			idxInBatch = 0
		}
	}
	if idxInBatch > 0 {
		nodeBatch = nodeBatch[:idxInBatch]
		chNodes <- nodeBatch
	}
	close(chNodes)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}
	fmt.Println("pair frequencies counted")

	// --------------------------------------------------------------------------------------------
	// step 2: compute conditional probability
	condProb := map[string]map[string]float64{}
	chKeys := make(chan []string)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func(idxWorker int) {
			numProcessedKeys := 0
			myRows := map[string]map[string]float64{}
			for keys := range chKeys {
				for _, text1 := range keys {
					rowFreq, _ := pairFreq[text1]
					sumRowFreq := 0.0
					for _, freq := range rowFreq {
						sumRowFreq += float64(freq)
					}
					rowCondProb := map[string]float64{}
					for text2, freq := range rowFreq {
						rowCondProb[text2] = float64(freq) / sumRowFreq
					}
					myRows[text1] = rowCondProb
				}
				numProcessedKeys += len(keys)
				fmt.Printf("worker %d has processed %d keys\n", idxWorker, numProcessedKeys)
			}
			lock.Lock()
			defer lock.Unlock()
			for key, row := range myRows {
				condProb[key] = row
			}
			fmt.Printf("worker %d exits\n", idxWorker)
			chWorkers <- true
		}(idxCPU)
	}

	keyBatch := make([]string, 10000)
	idxInBatch = 0
	for key, _ := range pairFreq {
		keyBatch[idxInBatch] = key
		idxInBatch++
		if idxInBatch == len(keyBatch) {
			chKeys <- keyBatch
			keyBatch = make([]string, 10000)
			idxInBatch = 0
		}
	}
	if idxInBatch > 0 {
		keyBatch = keyBatch[:idxInBatch]
		chKeys <- keyBatch
	}
	close(chKeys)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		<-chWorkers
	}
	fmt.Println("conditional probabilities computed")

	// --------------------------------------------------------------------------------------------
	// step 3: compute the result
	result := map[string]map[string]float64{}
	for text1, row := range condProb {
		result[text1] = map[string]float64{}
		for text2, p := range row {
			result[text1][text2] = 0.5 * p
		}
	}
	for text1, row := range condProb {
		for text2, p := range row {
			result[text2][text1] += 0.5 * p
		}
	}
	// must force the diagonals to be 1.0
	for text, row := range result {
		row[text] = 1.0
	}
	return result
}

// =================================================================================================
// func (g *CitationGraph) ClusterByLDA
// brief description: cluster the main nodes of g with their titles and their reference titles using
//	using CGSLDA.
// input:
//	numTopics: number of topics
//	alpha, beta: the parameters of CGSLDA
//	numIters: number of iterations for LDA
// output:
//	for each main node, gives the likelihoods the node belonging to a cluster.
func (g *CitationGraph) ClusterByLDA(numTopics int, alpha, beta float32, numIters int,
) map[int64][]float32 {
	if numTopics <= 0 || alpha <= 0.0 || beta <= 0.0 || numIters <= 0 {
		log.Fatalln("all parameters of ClusterByLDA must be > 0")
	}
	// ---------------------------------------------------------------------------------------------
	// step 1: create corpus for the main nodes
	corpus := NewCorpus()

	// ---------------------------------------------------------------------------------------------
	// step 2: create goroutines to add documents to the corpuse
	numCPUs := runtime.NumCPU()
	chNodes := make(chan map[int64]*CitationNode)
	chPhrases := make(chan map[int64][]string)
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		go func() {
			phrasess := map[int64][]string{}
			for nodes := range chNodes {
				for nodeID, node := range nodes {
					// (2.1) create an empty phrase list
					phrases := []string{}

					// (2.2) extract all possible phrases in title and put them in the phrase list
					phraseCandidates := KeyphraseExtraction.ExtractKeyPhraseCandidates(node.Title)
					for _, candidate := range phraseCandidates {
						candidatePhrases := KeyphraseExtraction.GetAllPossiblePhrases(candidate)
						for _, phrase := range candidatePhrases {
							phrases = append(phrases, phrase)
						}
					}

					// (2.3) extract all possible phrases in reference titles and put them in the
					// phrase list
					for _, refID := range node.Refs {
						refNode := g.Nodes[refID]
						phraseCandidates = KeyphraseExtraction.ExtractKeyPhraseCandidates(refNode.Title)
						for _, candidate := range phraseCandidates {
							candidatePhrases := KeyphraseExtraction.GetAllPossiblePhrases(candidate)
							for _, phrase := range candidatePhrases {
								phrases = append(phrases, phrase)
							}
						}
					}

					// (2.4) record the phrases
					phrasess[nodeID] = phrases
				}
			}
			chPhrases <- phrasess
		}()
	}

	// ---------------------------------------------------------------------------------------------
	// step 3: put nodes into the input channel
	nodes := map[int64]*CitationNode{}
	for idx, id := range g.ToBeAnalyzed {
		// (1.1) get a node
		node := g.Nodes[id]

		// (1.2) append nodes with this node
		nodes[id] = node

		// (1.3) send nodes when nodes are full
		if (idx+1)%len(nodes) == 0 || idx+1 == len(g.ToBeAnalyzed) {
			chNodes <- nodes
			nodes = map[int64]*CitationNode{}
		}
	}
	close(chNodes)

	// ---------------------------------------------------------------------------------------------
	// step 4: recieve phrases from the output channel
	phrasesOfNodes := map[int64][]string{}
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		phrasess := <-chPhrases
		for nodeID, phrases := range phrasess {
			phrasesOfNodes[nodeID] = phrases
		}
	}
	for _, phrases := range phrasesOfNodes {
		corpus.AddDoc(phrases)
	}

	// ---------------------------------------------------------------------------------------------
	// step 5: use the corpus to compute LDA
	LDA := NewCGSLDA(numTopics, alpha, beta, corpus)
	LDA.Train(numIters)

	// ---------------------------------------------------------------------------------------------
	// step 6: infer cluster memberships for each node
	memberships := map[int64][]float32{}
	for nodeID, phrases := range phrasesOfNodes {
		membershipsOfNode := LDA.Infer(phrases)
		memberships[nodeID] = membershipsOfNode
	}

	// ---------------------------------------------------------------------------------------------
	// step 7: return the result
	return memberships
}
