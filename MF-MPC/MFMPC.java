import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

// for Numerical Rating
public class MFMPC {
	// === Configurations
	// the number of latent dimensions
	public static int d = 20;
	public static int num_rating_types = 10; // !!! should be different for different data

	public static boolean flagGraded = true; // default value

	// tradeoff $\alpha_u$, $\alpha_v$, $\alpha_g$
	public static float alpha_u = 0.01f;
	public static float alpha_v = 0.01f;
	public static float alpha_g = 0.01f; // graded observation

	// tradeoff $\beta_u$, $\beta_v$
	public static float beta_u = 0.01f;
	public static float beta_v = 0.01f;

	// learning rate $\gamma$
	public static float gamma = 0.01f;

	// file names
	public static String fnTrainData = "";
	public static String fnTestData = "";

	//
	public static int n; // number of users
	public static int m; // number of items
	public static int num_train_target; // number of target training triples of (user,item,rating)
	public static int num_train_auxiliary; // number of auxiliary training triples of (user,item,rating)
	public static int num_train; // number of training triples of (user,item,rating) // num_train =
									// num_train_target+num_train_auxiliary
	public static int num_test; // number of test triples of (user,item,rating)

	public static float MinRating = 0.5f; // minimum rating value (0.5 for ML10M, Flixter; 1 for Netflix)
	public static float MaxRating = 5.0f; // maximum rating value

	// scan number over the whole data
	public static int num_iterations = 100;

	public static HashMap<Integer, HashMap<Integer, HashSet<Integer>>> Train_ExplicitFeedbacksGraded = new HashMap<Integer, HashMap<Integer, HashSet<Integer>>>();

	// === training data (target data)
	public static int[] indexUserTrain; // start from index "0"
	public static int[] indexItemTrain;
	public static float[] ratingTrain;

	// === test data
	public static int[] indexUserTest;
	public static int[] indexItemTest;
	public static float[] ratingTest;

	// === some statistics
	public static float[] userRatingSumTrain; // start from index "1"
	public static float[] itemRatingSumTrain;
	public static int[] userRatingNumTrain;
	public static int[] itemRatingNumTrain;

	public static int[][] user_graded_rating_number;

	// === model parameters to learn
	public static float[][] U;
	public static float[][] V;

	public static float[][][] G;

	public static float g_avg; // global average rating $\mu$
	public static float[] biasU; // bias of user
	public static float[] biasV; // bias of item

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	public static void main(String[] args) throws Exception {
		// ------------------------------
		// === Read the configurations
		for (int k = 0; k < args.length; k++) {
			if (args[k].equals("-d"))
				d = Integer.parseInt(args[++k]);
			else if (args[k].equals("-alpha_u"))
				alpha_u = Float.parseFloat(args[++k]);
			else if (args[k].equals("-alpha_v"))
				alpha_v = Float.parseFloat(args[++k]);
			else if (args[k].equals("-alpha_g"))
				alpha_g = Float.parseFloat(args[++k]);
			else if (args[k].equals("-beta_u"))
				beta_u = Float.parseFloat(args[++k]);
			else if (args[k].equals("-beta_v"))
				beta_v = Float.parseFloat(args[++k]);
			else if (args[k].equals("-gamma"))
				gamma = Float.parseFloat(args[++k]);
			else if (args[k].equals("-fnTrainData"))
				fnTrainData = args[++k];
			else if (args[k].equals("-fnTestData"))
				fnTestData = args[++k];
			else if (args[k].equals("-MinRating"))
				MinRating = Float.parseFloat(args[++k]);
			else if (args[k].equals("-MaxRating"))
				MaxRating = Float.parseFloat(args[++k]);
			else if (args[k].equals("-n"))
				n = Integer.parseInt(args[++k]);
			else if (args[k].equals("-m"))
				m = Integer.parseInt(args[++k]);
			else if (args[k].equals("-num_iterations"))
				num_iterations = Integer.parseInt(args[++k]);
			else if (args[k].equals("-flagGraded"))
				flagGraded = args[++k].toLowerCase().equals("true");
		}
		// ------------------------------
		// System.out.println(Arrays.toString(args));
		System.out.println("d: " + Integer.toString(d));
		System.out.println("alpha_u: " + Float.toString(alpha_u));
		System.out.println("alpha_v: " + Float.toString(alpha_v));
		System.out.println("alpha_g: " + Float.toString(alpha_g));
		System.out.println("beta_u: " + Float.toString(beta_u));
		System.out.println("beta_v: " + Float.toString(beta_v));
		System.out.println("gamma: " + Float.toString(gamma));
		System.out.println("fnTrainData: " + fnTrainData);
		System.out.println("fnTestData: " + fnTestData);
		System.out.println("MinRating: " + Float.toString(MinRating));
		System.out.println("MaxRating: " + Float.toString(MaxRating));
		System.out.println("n: " + Integer.toString(n));
		System.out.println("m: " + Integer.toString(m));
		System.out.println("num_iterations: " + Integer.toString(num_iterations));
		System.out.println("flagGraded: " + Boolean.toString(flagGraded));

		// ------------------------------
		// === Locate memory for the data structure
		// --- some statistics
		userRatingSumTrain = new float[n + 1]; // start from index "1"
		itemRatingSumTrain = new float[m + 1];
		userRatingNumTrain = new int[n + 1];
		itemRatingNumTrain = new int[m + 1];

		user_graded_rating_number = new int[n + 1][num_rating_types + 1];

		// --- model parameters to learn
		U = new float[n + 1][d]; // start from index "1"
		V = new float[m + 1][d];
		G = new float[m + 1][num_rating_types + 1][d];

		g_avg = 0; // global average rating $\mu$
		biasU = new float[n + 1]; // bias of user
		biasV = new float[m + 1]; // bias of item

		// ------------------------------
		// === Step 1: Read data
		long TIME_START_READ_DATA = System.currentTimeMillis();
		readData(fnTrainData, fnTestData);
		long TIME_FINISH_READ_DATA = System.currentTimeMillis();
		System.out.println("Elapsed Time (read data):" +
				Float.toString((TIME_FINISH_READ_DATA - TIME_START_READ_DATA) / 1000F)
				+ "s");
		// ------------------------------

		// ------------------------------
		// === Step 2: Check the result of average filling
		System.out.print("Average Filling (mu):");
		test();
		// ------------------------------

		// ------------------------------
		// === Step 3: Initialization of U, W, V
		long TIME_START_INITIALIZATION = System.currentTimeMillis();
		initialize();
		long TIME_FINISH_INITIALIZATION = System.currentTimeMillis();
		System.out.println("Elapsed Time (initialization):" +
				Float.toString((TIME_FINISH_INITIALIZATION - TIME_START_INITIALIZATION) / 1000F)
				+ "s");
		// ------------------------------

		// ------------------------------
		// === Step 4: Training and prediction
		long TIME_START_TRAIN = System.currentTimeMillis();
		train();
		long TIME_FINISH_TRAIN = System.currentTimeMillis();
		System.out.println("Elapsed Time (training):" +
				Float.toString((TIME_FINISH_TRAIN - TIME_START_TRAIN) / 1000F)
				+ "s");
		// ------------------------------
	}

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	@SuppressWarnings("resource")
	public static void readData(String fnTrainData, String fnTestData) throws Exception {
		// ----------------------------------------------------
		// --- number of target training records
		num_train_target = 0;
		BufferedReader brTrain = new BufferedReader(new FileReader(fnTrainData));
		String line = null;
		while ((line = brTrain.readLine()) != null) {
			num_train_target += 1;
		}
		System.out.println("num_train_target: " + num_train_target);
		num_train = num_train_target;

		// --- number of test records
		num_test = 0;
		BufferedReader brTest = new BufferedReader(new FileReader(fnTestData));
		line = null;
		while ((line = brTest.readLine()) != null) {
			num_test += 1;
		}
		System.out.println("num_test: " + num_test);
		// ----------------------------------------------------

		// ----------------------------------------------------
		// --- Locate memory for the data structure
		// --- train data
		indexUserTrain = new int[num_train]; // start from index "0"
		indexItemTrain = new int[num_train];
		ratingTrain = new float[num_train];

		// --- test data
		indexUserTest = new int[num_test];
		indexItemTest = new int[num_test];
		ratingTest = new float[num_test];
		// ----------------------------------------------------

		// ----------------------------------------------------
		int id_case = 0;
		double ratingSum = 0;
		// ----------------------------------------------------
		// Training data: (userID,itemID,rating)
		brTrain = new BufferedReader(new FileReader(fnTrainData));
		line = null;
		while ((line = brTrain.readLine()) != null) {
			String[] terms = line.split("\\s+|,|;");
			int userID = Integer.parseInt(terms[0]);
			int itemID = Integer.parseInt(terms[1]);
			float rating = Float.parseFloat(terms[2]);
			indexUserTrain[id_case] = userID;
			indexItemTrain[id_case] = itemID;
			ratingTrain[id_case] = rating;
			id_case += 1;

			// ---
			userRatingSumTrain[userID] += rating;
			userRatingNumTrain[userID] += 1;
			itemRatingSumTrain[itemID] += rating;
			itemRatingNumTrain[itemID] += 1;

			//
			ratingSum += rating;

			//
			if (flagGraded) {
				int g = (int) (rating * 2); // !!! convert grade index to 1,2,...,10
				if (Train_ExplicitFeedbacksGraded.containsKey(userID)) {
					HashMap<Integer, HashSet<Integer>> g2itemSet = Train_ExplicitFeedbacksGraded.get(userID);
					if (g2itemSet.containsKey(g)) {
						HashSet<Integer> itemSet = g2itemSet.get(g);
						itemSet.add(itemID);
						g2itemSet.put(g, itemSet);
					} else {
						HashSet<Integer> itemSet = new HashSet<Integer>();
						itemSet.add(itemID);
						g2itemSet.put(g, itemSet);
					}
					Train_ExplicitFeedbacksGraded.put(userID, g2itemSet);
				} else {
					HashMap<Integer, HashSet<Integer>> g2itemSet = new HashMap<Integer, HashSet<Integer>>();
					HashSet<Integer> itemSet = new HashSet<Integer>();
					itemSet.add(itemID);
					g2itemSet.put(g, itemSet);
					Train_ExplicitFeedbacksGraded.put(userID, g2itemSet);
				}
				// ---
				user_graded_rating_number[userID][g] += 1;
			}
		}
		brTrain.close();
		System.out.println("Finished reading the target training data");

		g_avg = (float) (ratingSum / num_train_target);
		System.out.println("average rating value: " + Float.toString(g_avg));
		// ----------------------------------------------------

		// ----------------------------------------------------
		// Test data: (userID,itemID,rating)
		id_case = 0; // initialize it to zero
		brTest = new BufferedReader(new FileReader(fnTestData));
		line = null;
		while ((line = brTest.readLine()) != null) {
			String[] terms = line.split("\\s+|,|;");
			int userID = Integer.parseInt(terms[0]);
			int itemID = Integer.parseInt(terms[1]);
			float rating = Float.parseFloat(terms[2]);
			indexUserTest[id_case] = userID;
			indexItemTest[id_case] = itemID;
			ratingTest[id_case] = rating;
			id_case += 1;
		}
		brTest.close();
		System.out.println("Finished reading the target test data");
		// ----------------------------------------------------
	}

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	public static void initialize() {
		// ======================================================
		// --- initialization of U, V, P, N, G
		for (int u = 1; u < n + 1; u++) {
			for (int f = 0; f < d; f++) {
				U[u][f] = (float) ((Math.random() - 0.5) * 0.01);
			}
		}
		//
		for (int i = 1; i < m + 1; i++) {
			for (int f = 0; f < d; f++) {
				V[i][f] = (float) ((Math.random() - 0.5) * 0.01);
			}
		}

		// --- G
		if (flagGraded) {
			for (int i = 1; i < m + 1; i++) {
				for (int g = 1; g <= num_rating_types; g++) {
					for (int f = 0; f < d; f++) {
						G[i][g][f] = (float) ((Math.random() - 0.5) * 0.01);
					}
				}
			}
		}
		// ======================================================

		// ======================================================
		// --- initialization of biasU, biasV
		for (int u = 1; u < n + 1; u++) {
			if (userRatingNumTrain[u] > 0) {
				biasU[u] = (userRatingSumTrain[u] - g_avg * userRatingNumTrain[u]) / userRatingNumTrain[u];
			}
		}
		//
		for (int i = 1; i < m + 1; i++) {
			if (itemRatingNumTrain[i] > 0) {
				biasV[i] = (itemRatingSumTrain[i] - g_avg * itemRatingNumTrain[i]) / itemRatingNumTrain[i];
			}
		}
		// ======================================================
	}

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	public static void train() {
		for (int iter = 0; iter < num_iterations; iter++) {
			System.out.print("Iter:" + Integer.toString(iter) + "| ");

			test();
			// ====================================

			for (int iter_rand = 0; iter_rand < num_train; iter_rand++) {
				// ===========================================
				// --- random sampling one triple of (userID,itemID,rating)
				int rand_case = (int) Math.floor(Math.random() * num_train);
				// Math.random(): [0.0, 1.0)
				int userID = indexUserTrain[rand_case];
				int itemID = indexItemTrain[rand_case];
				float rating = ratingTrain[rand_case];
				// ===========================================

				// ===========================================
				float[] tilde_Uu_g = new float[d];
				float[] tilde_Uu = new float[d];

				if (flagGraded) {
					for (int g = 1; g <= num_rating_types; g++) {
						if (user_graded_rating_number[userID][g] > 0) {
							// ---
							HashSet<Integer> itemSet = Train_ExplicitFeedbacksGraded.get(userID).get(g);

							// ---
							float explicit_feedback_num_u_sqrt = 0;
							if (itemSet.contains(itemID)) {
								if (itemSet.size() > 1) {
									explicit_feedback_num_u_sqrt = (float) Math
											.sqrt(user_graded_rating_number[userID][g] - 1);
								}
							} else {
								explicit_feedback_num_u_sqrt = (float) Math.sqrt(user_graded_rating_number[userID][g]);
							}

							if (explicit_feedback_num_u_sqrt > 0) {
								// --- aggregation
								for (int i2 : itemSet) {
									if (i2 != itemID) {
										for (int f = 0; f < d; f++) {
											tilde_Uu_g[f] += G[i2][g][f];
										}
									}
								}

								// --- normalization
								for (int f = 0; f < d; f++) {
									tilde_Uu_g[f] = tilde_Uu_g[f] / explicit_feedback_num_u_sqrt;
									tilde_Uu[f] += tilde_Uu_g[f];
									tilde_Uu_g[f] = 0;
								}
							}
						}
					}
				}

				// -----------------------
				// prediction and error
				float pred = 0;
				float err = 0;
				for (int f = 0; f < d; f++) {
					pred += U[userID][f] * V[itemID][f] + tilde_Uu[f] * V[itemID][f];
				}
				pred += g_avg + biasU[userID] + biasV[itemID];
				err = rating - pred;

				// -----------------------
				// --- update \mu
				g_avg = g_avg - gamma * (-err);

				// --- biasU, biasV
				biasU[userID] = biasU[userID] - gamma * (-err + beta_u * biasU[userID]);
				biasV[itemID] = biasV[itemID] - gamma * (-err + beta_v * biasV[itemID]);

				// -----------------------
				// --- update U, V
				float[] V_before_update = new float[d];
				float[] U_before_update = new float[d];
				for (int f = 0; f < d; f++) {
					V_before_update[f] = V[itemID][f];
					U_before_update[f] = U[userID][f];

					float grad_U_f = -err * V[itemID][f] + alpha_u * U[userID][f];
					float grad_V_f = -err * (U[userID][f] + tilde_Uu[f]) + alpha_v * V[itemID][f];
					U[userID][f] = U[userID][f] - gamma * grad_U_f;
					V[itemID][f] = V[itemID][f] - gamma * grad_V_f;
				}

				// --- update G
				if (flagGraded) {
					for (int g = 1; g <= num_rating_types; g++) {
						if (user_graded_rating_number[userID][g] > 0) {
							// ---
							HashSet<Integer> itemSet = Train_ExplicitFeedbacksGraded.get(userID).get(g);

							// ---
							float explicit_feedback_num_u_sqrt = 0;
							if (itemSet.contains(itemID)) {
								if (itemSet.size() > 1) {
									explicit_feedback_num_u_sqrt = (float) Math
											.sqrt(user_graded_rating_number[userID][g] - 1);
								}
							} else {
								explicit_feedback_num_u_sqrt = (float) Math.sqrt(user_graded_rating_number[userID][g]);
							}

							if (explicit_feedback_num_u_sqrt > 0) {
								for (int i2 : itemSet) {
									if (i2 != itemID) {
										for (int f = 0; f < d; f++) {
											G[i2][g][f] = G[i2][g][f] -
													gamma * (-err * V_before_update[f] / explicit_feedback_num_u_sqrt
															+ alpha_g * G[i2][g][f]);
										}
									}
								}
							}
						}
					}
				}
				// -----------------------
				// ===========================================
			}
			gamma = gamma * 0.9f;
		}
	}

	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	public static void test() {
		// --- number of test cases
		float mae = 0;
		float rmse = 0;

		// ====================================================
		// --- for efficiency
		float[] tilde_Uu_g = new float[d];
		float[][] tilde_Uu = new float[n + 1][d];

		for (int userID = 1; userID <= n; userID++) {
			// ---
			if (flagGraded) {
				for (int g = 1; g <= num_rating_types; g++) {
					if (user_graded_rating_number[userID][g] > 0) {
						float explicit_feedback_num_u_sqrt = (float) Math.sqrt(user_graded_rating_number[userID][g]);

						// ---
						HashSet<Integer> itemSet = Train_ExplicitFeedbacksGraded.get(userID).get(g);
						for (int i2 : itemSet) {
							for (int f = 0; f < d; f++) {
								tilde_Uu_g[f] += G[i2][g][f];
							}
						}

						// --- normalization
						for (int f = 0; f < d; f++) {
							tilde_Uu[userID][f] += tilde_Uu_g[f] / explicit_feedback_num_u_sqrt;
							tilde_Uu_g[f] = 0;
						}
					}
				}
			}

		}
		// ====================================================

		for (int t = 0; t < num_test; t++) {
			int userID = indexUserTest[t];
			int itemID = indexItemTest[t];
			float rating = ratingTest[t];

			// ===========================================
			// --- prediction via inner product
			float pred = g_avg + biasU[userID] + biasV[itemID];

			for (int f = 0; f < d; f++) {
				pred += U[userID][f] * V[itemID][f] + tilde_Uu[userID][f] * V[itemID][f];
			}
			// ===========================================

			// ===========================================
			// --- post processing predicted rating
			// if(pred < 1) pred = 1;
			// if(pred < 0.5) pred = 0.5f;
			// if(pred > 5) pred = 5;
			if (pred < MinRating)
				pred = MinRating;
			if (pred > MaxRating)
				pred = MaxRating;

			float err = pred - rating;
			mae += Math.abs(err);
			rmse += err * err;
			// ===========================================
		}
		float MAE = mae / num_test;
		float RMSE = (float) Math.sqrt(rmse / num_test);

		String result = "MAE:" + Float.toString(MAE) + "| RMSE:" + Float.toString(RMSE);
		System.out.println(result);
	}
	// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++

}