import java.util.Map;
import java.util.TreeMap;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class NaiveBayesImpl {
	int labelIndex;
	Map<Integer,Integer> lCount;
	Map<String,Integer> counts;
	Map<Integer,String> predictedLabels;
	Map<Integer,Double> condProb;
	Attribute label;
	Instances trainSet;
	Instances testSet;
	int numAttributes;
	int size, testSize;
	String predictedLabel;
	public NaiveBayesImpl(Instances trainSet, Instances testSet) {
		this.trainSet = trainSet;
		this.testSet = testSet;
		this.labelIndex = trainSet.numAttributes()-1;
		this.lCount = new TreeMap<Integer,Integer>();
		this.counts = new TreeMap<String,Integer>();
		this.predictedLabels = new TreeMap<Integer,String>();
		this.condProb = new TreeMap<Integer,Double>();
		this.label = trainSet.attribute(labelIndex);
		this.numAttributes = trainSet.numAttributes()-1;
		this.size = trainSet.numInstances();
		this.testSize = testSet.numInstances();
	}
	public void print() {
		for (int i = 0; i < testSet.numAttributes()-1;i++) {
			System.out.println(testSet.attribute(i).name() + " " + "class");
		}
		System.out.print("\n");
		classify();
		int count = 0;
		for (int i = 0; i < testSize; i++) {
			String prediction = predictedLabels.get(i);
			String actual = label.value((int)testSet.get(i).value(labelIndex));
			String temp = String.format("%.12f", condProb.get(i));
			while (temp.endsWith("0"))
				temp = temp.substring(0, temp.length()-1);
			System.out.println(prediction+" "+actual+" "+temp);
			if (actual.equals(prediction))
				count++;
		}
		System.out.println("\n"+String.valueOf(count)+"\n");
		System.out.println("\n"+String.valueOf(testSize)+"\n");
	}
	private void classify() {
		countLabels();
		count();
		double pmax = Double.MIN_VALUE;

		for(int l = 0; l < testSize; l++) {
			double total = 0.0;
			Instance instance = testSet.get(l);
			for (int i = 0; i < label.numValues(); i++) {
				int count = lCount.get(i);
				double p = Math.log((double)(count+1)/(size+label.numValues()));
				for(int j = 0; j < numAttributes; j++){
					Attribute att = trainSet.attribute(j);
					String name = att.name();
					for(int k = 0; k < att.numValues(); k++) {
						if((int)instance.value(att) == k) {
							String str = name+","+String.valueOf(k)+","+String.valueOf(i);
							if(counts.containsKey(str)) {
								double temp = (double)(counts.get(str)+1)/(count+att.numValues());
								p += Math.log(temp);
							}
							else {
								double temp = (double)1/(count+att.numValues());
								p += Math.log(temp);
							}
							break;
						}
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
			condProb.put(l, pmax/total);
			pmax = Double.MIN_VALUE;
		}
	}

	private void count() {
		for (int i = 0; i < size; i++) {
			Instance instance = trainSet.get(i);
			for (int j = 0; j < label.numValues(); j++) {
				if ((int)instance.value(label) == j) {
					for(int k = 0; k < numAttributes; k++) {
						Attribute att = trainSet.attribute(k);
						String name = att.name();
						for(int l = 0; l < att.numValues(); l++) {
							if((int)instance.value(att) == l) {
								String str = name+","+String.valueOf(l)+","+String.valueOf(j);
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