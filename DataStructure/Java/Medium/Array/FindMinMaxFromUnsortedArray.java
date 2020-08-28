package com.array;
public class FindMinMaxFromUnsortedArray {
	public static void main(String str[]) {
		int min = 0, max = 0;
		int[] arr = {5, 8, 2, 6, 10, 12, 3, 7, 34, 9};
		int n = arr.length;
		
		//Find length of array is odd or even and accordingly set min and max values
		if(n % 2 != 0) {
			min = max = arr[0];
		} else {
			if(arr[0] < arr[1]) {
				min = arr[0];
				max = arr[1];
			} else {
				min = arr[1];
				max = arr[0];
			}
		}
		
		int small, big;
		
		//Run a loop of n/2 times
		for(int i = 0; i < n; i = i + 2) {
			//1st comparison
			if(arr[i] < arr[i+1]) {
				small = arr[i];
				big = arr[i+1];
			} else {
				small = arr[i+1];
				big = arr[i];
			}
			
			//2nd comparison
			if(small < min) 
				min = small;
			
			//3rd comparison
			if(big > max)
				max = big;
		}
		
		System.out.println("The minimum value present in unsorted array is : " + min);
		System.out.println("The maximum value present in unsorted array is : " + max);
	}
}