// CitationGraphs.go project CitationGraphs.go
package CitationGraphs

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"

	"github.com/wujunfeng1/KeyphraseExtraction"
)

var reUnicodeHex *regexp.Regexp
var reUnicodeDec *regexp.Regexp

func init() {
	reUnicodeHex = regexp.MustCompile("&#[Xx]([A-Fa-f0-9])+;")
	reUnicodeDec = regexp.MustCompile("&#([0-9])+;")
}

/*
# ===========================================================================================
# struct CitationNode
# brief description: The node structure of a citation graph, the data of which are collected
#                    from crawling the website https://academic.microsoft.com , which is the
#                    public service website of MAG(Microsoft Academic Graph)
# fields:
#   id: The MAG(Microsoft Academic Graph) id of the paper at this node.
#       We can access the detail of the paper with the id by navigating the web link:
#       https://academic.microsoft.com/paper/$id
#   year: The year of publication of this paper.
#   title: The title of the paper.
#   labels: The labels from MAG(Microsoft Academic Graph).
#   refs: The references of this paper collected from MAG.
#         Please note that this field could be inaccurate: MAG suffers delay of info.
#         Many new papers with refs don't have refs listed in MAG.
#   cites: The citations to this paper from other papers (also collected from MAG).
#         Please note that this field is as inaccurate as refs due to the same reason.
*/
type CitationNode struct {
	ID     int64
	Year   int64
	Title  string
	Labels []string
	Refs   []int64
	Cites  []int64
}

/*
# ===========================================================================================
# struct CitationGraph
# brief description: A data structure of citation graph, the data of which are collected from
#                    the website https://academic.microsoft.com , which is the public service
#                    website of MAG(Microsoft Academic Graph)
# fields:
#   nodes: The node of struct CitationNode stored in a dictionary with key = id
#   toBeAnalyzed: The list of nodes to be analyzed. In ijcai_clustering dataset, these nodes
#                 are those published in IJCAI.
*/
type CitationGraph struct {
	Nodes        map[int64]*CitationNode
	ToBeAnalyzed []int64
}

/*
# ===========================================================================================
# function convertUnicodeHex
# brief description:
# 	Convert a hex representation string of unicode to the unicode.
# input:
#	s: The hex representation.
# output:
#	The unicode.
*/
func convertUnicodeHex(s []byte) []byte {
	ss := s[3 : len(s)-1]
	u, err := strconv.ParseInt(string(ss), 16, 64)
	if err != nil {
		return []byte("<?>")
	} else {
		return []byte(fmt.Sprintf("%c", u))
	}
}

/*
# ===========================================================================================
# function convertUnicodeDec
# brief description:
# 	Convert a dec representation string of unicode to the unicode.
# input:
#	s: The dec representation.
# output:
#	The unicode.
*/
func convertUnicodeDec(s []byte) []byte {
	ss := s[2 : len(s)-1]
	u, err := strconv.ParseInt(string(ss), 10, 64)
	if err != nil {
		return []byte("<?>")
	} else {
		return []byte(fmt.Sprintf("%c", u))
	}
}

/*
# ===========================================================================================
# function TidyTitle
# brief description: Make a title more tidy before using it. The procedure include the
#                    steps:
#                    (1) remove the spaces at the head and the tail of the title,
#                    (2) replace "&lt;" with "<",
#                    (3) replace "&gt;" with ">",
#                    (4) replace "&amp;" with "&",
#                    (5) replace "&quot;" with "\"",
#                    (6) replace "&apos;" with "'",
#                    (7) replace "&#[number];" with a corresponding unicode
# input:
#   title: The String (text) of a title.
# output:
#   A string of the tidied-up title.
*/
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
	// step 7: replace "&#[number];" with a corresponding unicode
	byteResult := []byte(result)
	byteResult = reUnicodeHex.ReplaceAllFunc(byteResult, convertUnicodeHex)
	byteResult = reUnicodeDec.ReplaceAllFunc(byteResult, convertUnicodeDec)
	result = string(byteResult)

	// --------------------------------------------------------------------------------------
	// step 8: return the result
	return result
}

/*
# ===========================================================================================
# function LoadCitationGraph
# brief description:
#	Load data from three files (nodes, edges, labels) of a citation graph.
# input:
#   path: The name of the path where the three files are stored.
#   prefix: The prefix of the names of the three files. For example, the prefix of
#           ijcai-citation-graph-nodes.csv is ijcai.
# output:
#    The citation graph represented by the three files.
*/
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

/*
# ===========================================================================================
# function SaveCitationGraph
# brief description: Save data to three files (nodes, edges, labels) of a citation graph.
# input:
#   path: The name of the path where the three files are stored.
#   prefix: The prefix of the names of the three files. For example, the prefix of
#           ijcai-citation-graph-nodes.csv is ijcai.
#   citationGraph: The citation graph represented by the three files.
# output:
#   nothing
*/
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

/*
# =================================================================================================
# method TFIDF
# brief description: compute the TFIDF of possible key phrases for each main node of a citationGraph
# input:
# 	nothing
# output:
#	the result of TFIDF grouped by the main nodes of the CitationGraph
*/
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

/*
# =================================================================================================
# method GetPhraseSimilarity
# brief description: compute the similarities between pairs of phrases and gives them in form of a
#	sparse matrix
# input:
# 	nothing
# output:
#	the sparse matrix of phrase similarities
*/
func (g *CitationGraph) GetPhraseSimilarity() map[string]map[string]float64 {
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
			for nodes := <-chNodes; len(nodes) > 0; nodes = <-chNodes {
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
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		chNodes <- []*CitationNode{}
	}
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
			for keys := <-chKeys; len(keys) > 0; keys = <-chKeys {
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
	for idxCPU := 0; idxCPU < numCPUs; idxCPU++ {
		chKeys <- []string{}
	}
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

/*
# =================================================================================================
# method SimTFIDF
# brief description: compute the Fuzzy TFIDF of possible key phrases for each main node of a citationGraph
# input:
# 	nothing
# output:
#	the result of Fuzzy TFIDF grouped by the main nodes of the CitationGraph
*/
func (g *CitationGraph) SimTFIDF() []map[string]float64 {
	candidateTFGroups := []map[string]float64{}
	phraseCandidateGroups := [][]string{}
	phraseSimilarity := g.GetPhraseSimilarity()
	fmt.Println("similarity generated")

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

/*
# =================================================================================================
# method SimTFSimIDF
# brief description: compute the Fuzzy TFIDF of possible key phrases for each main node of a citationGraph
# input:
# 	nothing
# output:
#	the result of Fuzzy TFIDF grouped by the main nodes of the CitationGraph
*/
func (g *CitationGraph) SimTFSimIDF() []map[string]float64 {
	candidateTFGroups := []map[string]float64{}
	phraseCandidateGroups := [][]string{}
	phraseSimilarity := g.GetPhraseSimilarity()
	fmt.Println("similarity generated")

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
