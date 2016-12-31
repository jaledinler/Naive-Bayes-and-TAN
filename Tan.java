import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class Tan {
	Map<String,Integer> counts;
	Map<String,Double> weights;
	Map<Integer,Integer> directed;
	Map<Integer,Integer> lCount;
	Map<String,Double> mst;
	Map<Integer,String> predictedLabels;
	Map<Integer,Double> postProb;
	String predictedLabel;
	Attribute label;
	Instances trainSet;
	Instances testSet;
	int labelIndex, numAttributes;
	int size, testSize;
	public Tan(Instances trainSet, Instances testSet) {
		this.directed = new TreeMap<Integer,Integer>();
		this.mst = new TreeMap<String,Double>();
		this.counts = new TreeMap<String,Integer>();
		this.lCount = new TreeMap<Integer,Integer>();
		this.weights = new TreeMap<String,Double>();
		this.trainSet = trainSet;
		this.testSet = testSet;
		this.labelIndex = trainSet.numAttributes()-1;
		this.label = trainSet.attribute(labelIndex);
		this.numAttributes = trainSet.numAttributes()-1;
		this.size = trainSet.numInstances();
		this.testSize = testSet.numInstances();
		this.predictedLabels = new TreeMap<Integer,String>();
		this.postProb = new TreeMap<Integer,Double>();
		countLabels();
		count_xi_given_label();		
		count_xi_xj_given_label();
		calcWeights();
	}
	public void print() {
		classify();
		System.out.println(trainSet.attribute(0).name()+" class");
		Set<Map.Entry<Integer,Integer>> set = directed.entrySet();
		Iterator<Map.Entry<Integer,Integer>> itr = set.iterator();
		while (itr.hasNext()) {
			Map.Entry<Integer,Integer> nextEntry = itr.next();
			int parent = nextEntry.getKey();
			int child = nextEntry.getValue();
			System.out.println(trainSet.attribute(parent).name()+ 
					" "+trainSet.attribute(child).name()+" class");
		}
		
		System.out.print("\n");
		int count = 0;
		for (int i = 0; i < testSize; i++) {
			String prediction = predictedLabels.get(i);
			String actual = label.value((int)testSet.get(i).value(labelIndex));
			String temp = String.format("%.12f", postProb.get(i));
			while (temp.endsWith("0"))
				temp = temp.substring(0, temp.length()-1);
			System.out.println(prediction+" "+actual+" "+temp);
			if (actual.equals(prediction))
				count++;
		}
		System.out.println("\n"+String.valueOf(count)+"\n");
		System.out.println("\n"+String.valueOf(testSize)+"\n");
	}
	private void classify(){
		MST();
		double pmax = Double.MIN_VALUE;

		for(int l = 0; l < testSize; l++) {
			double total = 0.0;
			Instance instance = testSet.get(l);
			for (int i = 0; i < label.numValues(); i++) {
				double p = Math.log((double)(lCount.get(i)+1)/(size+label.numValues()));
				Set<Map.Entry<String,Double>> set = mst.entrySet();
				Iterator<Map.Entry<String,Double>> itr = set.iterator();
				while(itr.hasNext()) {
					Map.Entry<String,Double> nextEntry = itr.next();
					String temp = nextEntry.getKey();
					String[] strArr = temp.split(",");
					int index1 = Integer.parseInt(strArr[0]);
					Attribute att = trainSet.attribute(index1);
					int index2 = Integer.parseInt(strArr[1]);
					for(int k = 0; k < att.numValues(); k++) {
						if((int)instance.value(att) == k) {
							for(int m = 0; m < trainSet.attribute(index2).numValues(); m++) {
								if((int)instance.value(trainSet.attribute(index2)) == m) {
									String str = String.valueOf(index1)+","+String.valueOf(k)+","+String.valueOf(index2)+","+String.valueOf(m)+","+
											String.valueOf(i);
									String str1 = String.valueOf(index1)+","+String.valueOf(k)+","+String.valueOf(i);
									if (counts.containsKey(str) && counts.containsKey(str1))
										p += Math.log((double)(counts.get(str)+1)/(counts.get(str1)+trainSet.attribute(index2).numValues()));
									if (!counts.containsKey(str) && counts.containsKey(str1))
										p += Math.log((double)1/(counts.get(str1)+trainSet.attribute(index2).numValues()));
									if (!counts.containsKey(str) && !counts.containsKey(str1))
										p += Math.log((double)1/trainSet.attribute(index2).numValues());
								}
							}
						}
					}
				}
				for(int m = 0; m < trainSet.attribute(0).numValues(); m++) {
					if((int)instance.value(trainSet.attribute(0)) == m) {
						String str = String.valueOf(0)+","+String.valueOf(m)+","+String.valueOf(i);
						p += Math.log((double)(counts.get(str)+1)/(lCount.get(i)+trainSet.attribute(0).numValues()));
					}
				}
				p = Math.exp(p);
				total += p;
				if (p > pmax) {
					pmax = p;
					predictedLabel = label.value(i);
				}
			}
			predictedLabels.put(l,predictedLabel);
			postProb.put(l, pmax/total);
			pmax = Double.MIN_VALUE;			
		}		
	}

	private void MST(){
		double max = -Double.MAX_VALUE;
		List<Attribute> vertices = new LinkedList<Attribute>();
		Map<Integer,Double> cw = new TreeMap<Integer,Double>();
		for (int i = 0; i < numAttributes; i++) {
			cw.put(i, max);
		}
		while (vertices.size() < numAttributes){
			max = -Double.MAX_VALUE;
			int index = 0;
			if (vertices.size() == 0) {
				index = 0;
			}
			else {
				double maxcv = -Double.MAX_VALUE;
				for (int i = 0; i < numAttributes;i++) {
					if (!vertices.contains(trainSet.attribute(i))){
						double weight = cw.get(i);
						if (weight > maxcv) {
							maxcv = weight;
							index = i;
						}
					}
				}
				String str = String.valueOf(directed.get(index));
				mst.put(str+","+String.valueOf(index),maxcv);
			}
			Attribute v = trainSet.attribute(index);
			vertices.add(v);			
			Set<Map.Entry<String,Double>> set = weights.entrySet();
			Iterator<Map.Entry<String,Double>> itr = set.iterator();
			String str = null;
			int indexw = 0;
			while (itr.hasNext()) {
				Map.Entry<String,Double> nextEntry = itr.next();
				str = nextEntry.getKey();
				String[] strArr = str.split(",");				
				if (strArr[0].equals(String.valueOf(index)) && !vertices.contains(trainSet.attribute(Integer.parseInt(strArr[1])))) {
					double weight = nextEntry.getValue();
					if (weight > cw.get(Integer.parseInt(strArr[1]))) {
						indexw = Integer.parseInt(strArr[1]);
						cw.put(indexw, weight);
						directed.put(indexw, index);
					}
				}
			}			
		}
	}
	private void calcWeights() {
		for(int l = 0; l < numAttributes; l++) {
			for(int i = 0 ; i < numAttributes; i++) {
				if(i == l)
					continue;				
				calcWeight(l,i);

			}
		}
	}
	private void calcWeight(int att1, int att2) {
		double sum = 0.0;
		int val1 = trainSet.attribute(att1).numValues();
		for (int i = 0; i < val1; i++) {
			int val2 = trainSet.attribute(att2).numValues();
			for (int j = 0; j < val2; j++) {
				for (int l = 0; l < label.numValues(); l++) {
					String str = String.valueOf(att1)+","+String.valueOf(i)+","+String.valueOf(att2)+","+
							String.valueOf(j)+","+String.valueOf(l);
					String str1 = String.valueOf(att1)+","+String.valueOf(i)+","+String.valueOf(l);
					String str2 = String.valueOf(att2)+","+String.valueOf(j)+","+String.valueOf(l);					
					
					double allProb = 0.0;
					if (counts.containsKey(str))
						allProb = (double)(counts.get(str)+1)/(size+val1*val2*label.numValues());
					else 
						allProb = (double)1/(size+val1*val2*label.numValues());
					
					double condProb2 = 0.0;
					if (counts.containsKey(str1))
						condProb2 = (double)(counts.get(str1)+1)/(lCount.get(l)+val1);
					else 
						condProb2 = (double)1/(lCount.get(l)+val1);
					
					double condProb3 = 0.0;
					if (counts.containsKey(str2))
						condProb3 = (double)(counts.get(str2)+1)/(lCount.get(l)+val2);
					else 
						condProb3 = (double)1/(lCount.get(l)+val2);
					
					double condProb1 = 0.0;
					
					if(counts.containsKey(str))
						condProb1 = (double)(counts.get(str)+1)/(lCount.get(l)+val1*val2);
					else
						condProb1 = (double)1/(lCount.get(l)+val1*val2);
					double temp = allProb*(Math.log(condProb1/(condProb2*condProb3))/Math.log(10)); 
					sum += temp;
				}
			}
		}
		weights.put(String.valueOf(att1)+","+String.valueOf(att2), sum);
	}
	
	private void count_xi_xj_given_label() {
		for (int i = 0; i < size; i++) {
			Instance instance = trainSet.get(i);
			for (int j = 0; j < label.numValues(); j++) {
				if ((int)instance.value(label) == j) {
					for(int k = 0; k < numAttributes; k++) {
						Attribute att = trainSet.attribute(k);
						for(int l = 0; l < att.numValues(); l++) {
							if((int)instance.value(att) == l) {
								for(int m = 0; m < numAttributes; m++) {
									if (m == k)
										continue;
									Attribute att2 = trainSet.attribute(m);
									for(int n = 0; n < att2.numValues(); n++) {
										if((int)instance.value(att2) == n) {
											String str = String.valueOf(k)+","+String.valueOf(l)+","+
													String.valueOf(m)+","+String.valueOf(n)+","+String.valueOf(j);
											if(counts.containsKey(str))
												counts.put(str, counts.get(str)+1);
											else
												counts.put(str,1);
											break;
										}
									}
								}
							}
						}                       
					}
				}
			}			
		}
	}
	private void count_xi_given_label() {
		for (int i = 0; i < size; i++) {
			Instance instance = trainSet.get(i);
			for (int j = 0; j < label.numValues(); j++) {
				if ((int)instance.value(label) == j) {
					for(int k = 0; k < numAttributes; k++) {
						Attribute att = trainSet.attribute(k);
						for(int l = 0; l < att.numValues(); l++) {
							if((int)instance.value(att) == l) {
								String str = String.valueOf(k)+","+String.valueOf(l)+","+String.valueOf(j);
								if(counts.containsKey(str))
									counts.put(str, counts.get(str)+1);
								else
									counts.put(str,1);
								break;
							}
						}                       
					}
				}
			}			
		}
	}
	private void countLabels() {
		for(int i = 0; i < size; i++) {
			Instance instance = trainSet.get(i);
			for (int j = 0; j < label.numValues(); j++) {
				if ((int)instance.value(label) == j) {
					if (lCount.containsKey(j))
						lCount.put(j, lCount.get(j)+1);
					else
						lCount.put(j, 1);
				}
			}
		}
	}
}