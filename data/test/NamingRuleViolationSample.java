package com.project.core.util;

public class DataProcessingUtil { 

    private static final int defaultTimeout = 60;

    public String GetUserData(int id) { 
        String strName = "Guest";
        if (id > 100) {
            strName = "VIP";
        }
        return strName;
    }

    public double calTotal(List<Item> items) {
        double currentTotal = 0.0;
        
        for (int i=0; i < items.size(); i++) {
            currentTotal += items.get(i).price;
        }
        return currentTotal;
    }
    
    private List<User> user_id_list = new ArrayList<>(); 
    
    private int iCount = 0; 
    
    private String 배송지주소 = "서울";

}